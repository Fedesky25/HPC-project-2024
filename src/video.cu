//
// Created by feder on 27/09/2024.
//

#include "video.cuh"
#include "frames.cuh"
#include <fstream>
#include <iomanip>
#include <omp.h>

extern "C" {
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}


#define HANDLE_AV_ERROR(EXPR, MSG) { \
    auto errnum = (EXPR);       \
    if(errnum < 0) {            \
        char reason[AV_ERROR_MAX_STRING_SIZE] = {0}; \
        av_strerror(errnum, reason, AV_ERROR_MAX_STRING_SIZE); \
        std::cerr << MSG << ": " << reason << std::endl; \
        exit(1);                    \
    }                                \
}


#define PRINT_TIMES(TIME) {\
    std::cout << "   " << std::setw(5) << (TIME)                                  \
              << " | " << std::setw(5) << tc[0] << " | " << std::setw(5) << tw[0] \
              << " | " << std::setw(5) << tc[1] << " | " << std::setw(5) << tw[1] \
              << " | " << std::setw(5) << tc[2] << " | " << std::setw(5) << tw[2] \
              << " | " << std::setw(5) << tc[3] << " | " << std::setw(5) << tw[3] \
              << " | " << std::setw(5) << tc[4] << " | " << std::setw(5) << tw[4] \
              << " | " << std::setw(5) << tc[5] << " | " << std::setw(5) << tw[5] \
              << " | " << std::setw(5) << tc[6] << " | " << std::setw(5) << tw[6] \
              << " | " << std::setw(5) << tc[7] << " | " << std::setw(5) << tw[7] \
              << std::endl;                                                       \
    tc[0] = tc[1] = tc[2] = tc[3] = tc[4] = tc[5] = tc[6] = tc[7] = 0;            \
    tw[0] = tw[1] = tw[2] = tw[3] = tw[4] = tw[5] = tw[6] = tw[7] = 0;            \
    }


#define PRINT_SUMMARY(FACTOR) { \
    auto m = 0.1f / total;                    \
    auto writing = tw[0] + tw[1] + tw[2] + tw[3] + tw[4] + tw[5] + tw[6] + tw[7];     \
    auto computation = tc[0] + tc[1] + tc[2] + tc[3] + tc[4] + tc[5] + tc[6] + tc[7]; \
    std::cout << frame_count << " frames written in " << std::setprecision(3) << total << "s (computation: "  \
              << std::fixed << std::setprecision(2) << computation*m*(FACTOR) << "%, file write: "  \
              << std::fixed << std::setprecision(2) << writing*m << "%)" << std::endl; \
}


struct StreamWrapper {
    const AVCodec * codec;
    AVStream * stream;
    AVPacket * packet;
    /// Encoding context
    AVCodecContext * enc_ctx;
    /// Format context
    AVFormatContext * fmt_ctx;

    void open(const Configuration& config, bool opaque) {
        packet = av_packet_alloc();
        EXIT_IF(!packet, "Could not allocate AV packet")

        avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, config.output);
        if(!fmt_ctx || fmt_ctx->oformat->video_codec == AV_CODEC_ID_NONE) {
            std::cout << "The output format could not be deduced from file extension"
                      << " or does not support video output: fallback to MPEG" << std::endl;
            avformat_alloc_output_context2(&fmt_ctx, nullptr, "mpeg", config.output);
            EXIT_IF(!fmt_ctx, "Could not open MPEG output context")
        }

        codec = avcodec_find_encoder(fmt_ctx->oformat->video_codec);
        EXIT_IF(!codec, "Could not find encoder for" << avcodec_get_name(fmt_ctx->oformat->video_codec))

        stream = avformat_new_stream(fmt_ctx, codec);
        EXIT_IF(!stream, "Could not allocate video stream")
        stream->id = 0;
        stream->time_base = AVRational{1, config.evolution.frame_rate};

        enc_ctx = avcodec_alloc_context3(codec);
        EXIT_IF(!enc_ctx, "Could not allocate encoding context")
        enc_ctx->bit_rate = 400000;
        enc_ctx->codec_id = fmt_ctx->oformat->video_codec;
        enc_ctx->width = config.canvas.width;
        enc_ctx->height = config.canvas.height;
        enc_ctx->time_base = AVRational{1, config.evolution.frame_rate};
        enc_ctx->framerate = AVRational{config.evolution.frame_rate, 1};
        enc_ctx->gop_size = 12;
        enc_ctx->pix_fmt = opaque ? AV_PIX_FMT_YUVJ444P : AV_PIX_FMT_YUVA444P;
        // if (codec->id == AV_CODEC_ID_H264) av_opt_set(enc_ctx->priv_data, "preset", "slow", 0);
        if (enc_ctx->codec_id == AV_CODEC_ID_MPEG2VIDEO) enc_ctx->max_b_frames = 2;
        if (enc_ctx->codec_id == AV_CODEC_ID_MPEG1VIDEO) enc_ctx->mb_decision = 2;
        if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) enc_ctx->flags |= AVFMT_GLOBALHEADER;

        HANDLE_AV_ERROR(avcodec_open2(enc_ctx, codec, nullptr), "Could not open video codec")
        HANDLE_AV_ERROR(avcodec_parameters_from_context(stream->codecpar, enc_ctx), "Could not copy the stream parameters")

        av_dump_format(fmt_ctx, 0, config.output, 1);
        HANDLE_AV_ERROR(avio_open(&(fmt_ctx->pb), config.output, AVIO_FLAG_WRITE), "Could not open output file")
        HANDLE_AV_ERROR(avformat_write_header(fmt_ctx, nullptr), "Could not write headers")
    }

    AVFrame * get_frame() const {
        auto frame = av_frame_alloc();
        EXIT_IF(!frame, "Could not allocate video frame")
        frame->format = enc_ctx->pix_fmt;
        frame->width  = enc_ctx->width;
        frame->height = enc_ctx->height;
        HANDLE_AV_ERROR(av_frame_get_buffer(frame, 0), "Could not allocate the video frame buffer")
        return frame;
    }

    void encode(AVFrame * frame) const {
        HANDLE_AV_ERROR(avcodec_send_frame(enc_ctx, frame), "Could not send frame")
        int ret;
        do {
            ret = avcodec_receive_packet(enc_ctx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                return;
            else if (ret < 0) {
                char reason[AV_ERROR_MAX_STRING_SIZE] = {0};
                av_strerror(ret, reason, AV_ERROR_MAX_STRING_SIZE);
                std::cerr << "Error during encoding: " << reason << std::endl;
                exit(1);
            }
            av_packet_rescale_ts(packet, enc_ctx->time_base, stream->time_base);
            packet->stream_index = stream->index;
            av_interleaved_write_frame(fmt_ctx, packet);
//            av_write_frame(fmt_ctx, packet);
//            av_packet_unref(packet);
        } while (ret >= 0);
    }

    void close() {
        encode(nullptr);
        av_write_trailer(fmt_ctx);
        avcodec_free_context(&enc_ctx);
        av_packet_free(&packet);
        avio_closep(&(fmt_ctx->pb));
        avformat_free_context(fmt_ctx);
    }
};

auto get_ctx(int width, int height, int frame_rate, bool opaque) {
    auto codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    EXIT_IF(!codec, "Could not find H264 codec")
    auto ctx = avcodec_alloc_context3(codec);
    EXIT_IF(!ctx, "Could not allocate codec context")
    ctx->bit_rate = 400000;
    ctx->width = width;
    ctx->height = height;
    ctx->time_base = AVRational{1, frame_rate};
    ctx->framerate = AVRational{frame_rate, 1};
    ctx->pix_fmt = opaque ? AV_PIX_FMT_YUVJ444P : AV_PIX_FMT_YUVA444P;
//    if (codec->id == AV_CODEC_ID_H264)
    av_opt_set(ctx->priv_data, "preset", "slow", 0);
    HANDLE_AV_ERROR(avcodec_open2(ctx, codec, nullptr), "Could not open codec")
    return ctx;
}

auto get_frame(AVCodecContext * ctx) {
    auto frame = av_frame_alloc();
    EXIT_IF(!frame, "Could not allocate video frame")
    frame->format = ctx->pix_fmt;
    frame->width  = ctx->width;
    frame->height = ctx->height;
    HANDLE_AV_ERROR(av_frame_get_buffer(frame, 0), "Could not allocate the video frame buffer");
    return frame;
}

void encode(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt, FILE *outfile) {
    HANDLE_AV_ERROR(avcodec_send_frame(enc_ctx, frame), "Could not send frame")
    int ret;
    do {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            char reason[AV_ERROR_MAX_STRING_SIZE] = {0};
            av_strerror(ret, reason, AV_ERROR_MAX_STRING_SIZE);
            std::cerr << "Error during encoding: " << reason << std::endl;
            exit(1);
        }
        fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(pkt);
    } while (ret >= 0);
}


template<bool opaque>
void write_video_serial_internal(const Configuration & config, Canvas canvas) {
    StreamWrapper w = { nullptr };
    w.open(config, opaque);
    auto frame = w.get_frame();

    auto frame_count = config.evolution.frame_count;
    float tc[8] = {0}, tw[8] = {0};
    if(verbose) std::cout << "Frame computation (iteration, (computation [ms], writing [ms]) * 8):" << std::endl << std::setprecision(1);
    timers(2)
    tick(0)
    for(int32_t t=0; t<frame_count; t++) {
        HANDLE_AV_ERROR(av_frame_make_writable(frame), "Frame cannot be written")
        frame->pts = t;
        tick(1)
        compute_frame_serial<opaque>(t, frame_count, config.evolution.life_time, canvas, frame, &config.background);
        tock_ms(1)
        tc[t&7] += t_elapsed;
        tick(1)
        w.encode(frame);
        tock_ms(1)
        tw[t&7] += t_elapsed;
        if(verbose && (t&7) == 7) PRINT_TIMES(t+1)
    }
    tock_s(0)
    auto total = t_elapsed;
    auto remaining = (frame_count-1)&7;
    if(verbose) {
        if(remaining) {
            std::cout << "   " << std::setw(5) << frame_count
                      << " | " << std::setw(5) << tc[0] << " | " << std::setw(5) << tw[0];
            for(int32_t j=1; j<remaining; j++)
                std::cout << " | " << std::setw(5) << tc[j] << " | " << std::setw(5) << tw[j];
            std::cout << std::endl;
        }
        std::cout << "  :: total " << total << 's' << std::endl;
    }
    else PRINT_SUMMARY(1)
    av_frame_free(&frame);
    w.close();
}

void write_video_serial(const Configuration & config, Canvas canvas) {
    if(config.background.A == 1.0f) write_video_serial_internal<true>(config, canvas);
    else write_video_serial_internal<false>(config, canvas);
}


template<bool opaque>
void write_video_omp_internal(const Configuration & config, const Canvas * canvases, uint32_t canvas_count) {
    StreamWrapper w = {nullptr};
    w.open(config, opaque);
    AVFrame * frame_buffers[2] = { w.get_frame(), w.get_frame() };

    auto frame_count = config.evolution.frame_count;
    auto life_time = config.evolution.life_time;
    float tc[8] = {0}, tw[8] = {0};
    tw[0] = -1.0;

    omp_set_nested(1);

    if(verbose) std::cout << "Frame computation (iteration, (computation [ms], writing [ms]) * 8):" << std::endl << std::setprecision(1);
    auto start_all = std::chrono::steady_clock::now();

    compute_frame_omp<opaque>(0, frame_count, life_time, canvases, canvas_count, frame_buffers[0], &config.background);

    for(int32_t t=1; t<frame_count; t++) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                auto start = std::chrono::steady_clock::now();
                w.encode(frame_buffers[(t-1)&1]);
                auto end = std::chrono::steady_clock::now();
                tw[t&7] += (std::chrono::duration<float, std::milli>(end-start)).count();
            }
            #pragma omp section
            {
                auto start = std::chrono::steady_clock::now();
                auto frame = frame_buffers[t&1];
                HANDLE_AV_ERROR(av_frame_make_writable(frame), "Frame cannot be written")
                compute_frame_omp<opaque>(t, frame_count, life_time, canvases, canvas_count, frame_buffers[t&1], &config.background);
                frame->pts = t;
                auto end = std::chrono::steady_clock::now();
                tc[t&7] += (std::chrono::duration<float, std::milli>(end-start)).count();
            }
        }
        if(verbose && (t&7) == 7) PRINT_TIMES(t+1)
    }
    auto start = std::chrono::steady_clock::now();
    w.encode(frame_buffers[(frame_count-1)&1]);
    auto end = std::chrono::steady_clock::now();
    tw[(frame_count-1)&7] += (std::chrono::duration<float, std::milli>(end-start)).count();
    auto end_all = std::chrono::steady_clock::now();
    float total = (std::chrono::duration<float, std::ratio<1>>(end_all-start_all)).count();

    if(verbose) std::cout << "  :: total " << total << 's' << std::endl;
    else PRINT_SUMMARY(1)

    av_frame_free(&frame_buffers[0]);
    av_frame_free(&frame_buffers[1]);
    w.close();
}


void write_video_omp(const Configuration & config, const Canvas * canvases, uint32_t canvas_count) {
    if(config.background.A == 1.0f) write_video_omp_internal<true>(config, canvases, canvas_count);
    else write_video_omp_internal<false>(config, canvases, canvas_count);
}


template<bool opaque>
void write_video_gpu_internal(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count, int32_t lifetime,
        const RGBA * background
) {
    std::ofstream raw_output(filename);
    auto frame_mem = frame_size * sizeof(uint32_t);
    unsigned char *h_frame, *d_frame[2];
    h_frame = (unsigned char*) malloc(frame_mem);
    cudaMalloc(d_frame, frame_mem);
    cudaMalloc(d_frame+1, frame_mem);
    if(verbose) {
        std::cout << "Frame buffers: CPU=" << (((frame_mem - 1) >> 20) + 1) << "MB, GPU="
                  << (((frame_mem * 2 - 1) >> 20) + 1) << "MB" << std::endl << std::fixed;
        std::cout << "Frame computation (iteration, (computation [us], writing [ms]) * 8):" << std::endl;
    }

    float tw[8] = {0}, tc[8] = {0};
    auto begin = std::chrono::steady_clock::now();
    compute_frame_gpu<opaque>(
            0, frame_count,
            canvases, canvas_count,
            d_frame[0], frame_size, lifetime,
            background);
    cudaDeviceSynchronize();
    auto _end = std::chrono::steady_clock::now();
    tc[0] += (std::chrono::duration<float,std::micro>(_end-begin)).count();
    auto start_all = begin;

    for(int32_t t=1; t < frame_count; t++) {
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                auto start = std::chrono::steady_clock::now();
                cudaMemcpy(h_frame, d_frame[(t & 1) ^ 1], frame_mem, cudaMemcpyDeviceToHost);
                raw_output.write(reinterpret_cast<const char *>(h_frame), frame_mem);
                auto end = std::chrono::steady_clock::now();
                tw[(t - 1) & 7] += (std::chrono::duration<float, std::milli>(end - start)).count();
            }
            #pragma omp section
            {
                auto start = std::chrono::steady_clock::now();
                compute_frame_gpu<opaque>(t, frame_count, canvases, canvas_count, d_frame[t & 1], frame_size, lifetime, background);
                cudaDeviceSynchronize();
                auto end = std::chrono::steady_clock::now();
                tc[t & 7] += (std::chrono::duration<float,std::micro>(end - start)).count();
            }
        }
        if(verbose && (t & 7) == 0) PRINT_TIMES(t)
    }
    begin = std::chrono::steady_clock::now();
    cudaMemcpy(h_frame, d_frame[(frame_count-1)&1], frame_mem, cudaMemcpyDeviceToHost);
    raw_output.write(reinterpret_cast<const char *>(h_frame), frame_mem);
    _end = std::chrono::steady_clock::now();
    tw[(frame_count-1)&7] += (std::chrono::duration<float,std::milli>(_end-begin)).count();
    if((frame_count & 7) == 0 && verbose) PRINT_TIMES(frame_count)
    _end = std::chrono::steady_clock::now();
    float total = (std::chrono::duration<float, std::ratio<1>>(_end-start_all)).count();
    if(verbose) std::cout << "  :: total " << total << 's' << std::endl;
    else PRINT_SUMMARY(1e-3f)

    cudaFree(d_frame[0]);
    cudaFree(d_frame[1]);
    free(h_frame);
}

void write_video_gpu(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count,
        int32_t lifetime, const RGBA * background
) {
    if(background->A == 1.0f) write_video_gpu_internal<true>(filename, canvases, canvas_count, frame_size, frame_count, lifetime, background);
    else write_video_gpu_internal<false>(filename, canvases, canvas_count, frame_size, frame_count, lifetime, background);
}