#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <stdio.h>
#include <numeric>
#include <map>
#include <iostream>
#include <algorithm>


#include <vector>
// ncnn

#include <sys/time.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <net.h>

#include "centerface.id.h"
#define NMS_UNION 1
#define NMS_MIN  2

struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float area;
    float landmarks[10];
};


static ncnn::Net net;
std::vector<FaceInfo> face_info;

//    int init(std::string model_path);


//You can change the shape of input image by setting params :resized_w and resized_h
//static int detect(ncnn::Mat &inblob, std::vector<FaceInfo>&faces, int resized_w,int resized_h,
//           float scoreThresh = 0.5, float nmsThresh = 0.3);
//
////private:
//static void dynamicScale(float in_w, float in_h);
//static void squareBox(std::vector<FaceInfo> &faces);
//static void genIds(float *heatmap, int h, int w, float thresh, std::vector<int> &ids);
//static nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, float nmsthreshold = 0.3,int type=NMS_MIN);
//decode(ncnn::Mat &heatmap, ncnn::Mat &scale, ncnn::Mat &offset, ncnn::Mat &landmarks,
//            std::vector<FaceInfo>&faces, float scoreThresh, float nmsThresh);
//private:


int d_h;
int d_w;
float d_scale_h;
float d_scale_w;

float scale_h;
float scale_w;

int image_h;
int image_w;
float scoreThresh=0.5;
float nmsThresh=0.3;
int type=2;








void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, float nmsthreshold=0.3,int type=2)
{
    if (input.empty()) {
        return;
    }
    std::sort(input.begin(), input.end(),
              [](const FaceInfo& a, const FaceInfo& b)
              {
                  return a.score < b.score;
              });

    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    std::vector<int> vPick;
    int nPick = 0;
    std::multimap<float, int> vScores;
    const int num_boxes = input.size();
    vPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i) {
        vScores.insert(std::pair<float, int>(input[i].score, i));
    }
    while (vScores.size() > 0) {
        int last = vScores.rbegin()->second;
        vPick[nPick] = last;
        nPick += 1;
        for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
            int it_idx = it->second;
            maxX = std::max(input.at(it_idx).x1, input.at(last).x1);
            maxY = std::max(input.at(it_idx).y1, input.at(last).y1);
            minX = std::min(input.at(it_idx).x2, input.at(last).x2);
            minY = std::min(input.at(it_idx).y2, input.at(last).y2);
            //maxX1 and maxY1 reuse
            maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
            //IOU reuse for the area of two bbox
            IOU = maxX * maxY;
            if (type==NMS_UNION)
                IOU = IOU / (input.at(it_idx).area + input.at(last).area - IOU);
            else if (type == NMS_MIN) {
                IOU = IOU / ((input.at(it_idx).area < input.at(last).area) ? input.at(it_idx).area : input.at(last).area);
            }
            if (IOU > nmsthreshold) {
                it = vScores.erase(it);
            }
            else {
                it++;
            }
        }
    }

    vPick.resize(nPick);
    output.resize(nPick);
    for (int i = 0; i < nPick; i++) {
        output[i] = input[vPick[i]];
    }
}

void genIds(float * heatmap, int h, int w, float thresh, std::vector<int>& ids)
{
    if (heatmap==NULL)
    {
        std::cout << "heatmap is nullptr,please check! " << std::endl;
        return;
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (heatmap[i*w + j] > thresh) {
                ids.push_back(i);
                ids.push_back(j);
            }
        }
    }
}

void decode(ncnn::Mat & heatmap, ncnn::Mat & scale, ncnn::Mat & offset, ncnn::Mat & landmarks, std::vector<FaceInfo>& faces, float scoreThresh=0.5, float nmsThresh=0.3)
{
    int fea_h = heatmap.h;
    int fea_w = heatmap.w;
    int spacial_size = fea_w*fea_h;

    float *heatmap_ = (float*)(heatmap.data);

    float *scale0 = (float*)(scale.data);
    float *scale1 = scale0 + spacial_size;

    float *offset0 = (float*)(offset.data);
    float *offset1 = offset0 + spacial_size;

    std::vector<int> ids;
    genIds(heatmap_, fea_h, fea_w, scoreThresh, ids);
//    face_info.size()
    __android_log_print(ANDROID_LOG_DEBUG, "centerface", "num of ids %d ",ids.size());
    std::vector<FaceInfo> faces_tmp;
    for (int i = 0; i < ids.size() / 2; i++) {
        int id_h = ids[2 * i];
        int id_w = ids[2 * i + 1];
        int index = id_h*fea_w + id_w;

        float s0 = std::exp(scale0[index]) * 4;
        float s1 = std::exp(scale1[index]) * 4;
        float o0 = offset0[index];
        float o1 = offset1[index];

        //std::cout << s0 << " " << s1 << " " << o0 << " " << o1 << std::endl;

        float x1 = (id_w + o1 + 0.5) * 4 - s1 / 2 > 0.f ? (id_w + o1 + 0.5) * 4 - s1 / 2 : 0;
        float y1 =(id_h + o0 + 0.5) * 4 - s0 / 2 > 0 ? (id_h + o0 + 0.5) * 4 - s0 / 2 : 0;
        float x2 = 0, y2 = 0;
        x1 = x1 < (float)d_w ? x1 : (float)d_w;
        y1 = y1 < (float)d_h ? y1 : (float)d_h;
        x2 =  x1 + s1 < (float)d_w ? x1 + s1 : (float)d_w;
        y2 = y1 + s0 < (float)d_h ? y1 + s0 : (float)d_h;

        //std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;

        FaceInfo facebox;
        facebox.x1 = x1;
        facebox.y1 = y1;
        facebox.x2 = x2;
        facebox.y2 = y2;
        facebox.score = heatmap_[index];
        facebox.area=(facebox.x2-facebox.x1)*(facebox.y2-facebox.y1);


        float box_w = x2 - x1; //=s1?
        float box_h = y2 - y1; //=s0?

        //std::cout << facebox.x1 << " " << facebox.y1 << " " << facebox.x2 << " " << facebox.y2 << std::endl;
        for (int j = 0; j < 5; j++) {
            float *xmap = (float*)landmarks.data + (2 * j + 1)*spacial_size;
            float *ymap = (float*)landmarks.data + (2 * j)*spacial_size;
            facebox.landmarks[2*j] = x1 + xmap[index] * s1;//box_w;
            facebox.landmarks[2 * j+1] = y1 + ymap[index] *  s0; // box_h;
        }
        faces_tmp.push_back(facebox);
    }
    __android_log_print(ANDROID_LOG_DEBUG, "centerface", "num of face before nms %d ",faces.size());
    nms(faces_tmp, faces, nmsThresh,2);
    __android_log_print(ANDROID_LOG_DEBUG, "centerface", "num of face after nms %d ",faces.size());
    for (int k = 0; k < faces.size(); k++) {
        faces[k].x1 *= d_scale_w*scale_w;
        faces[k].y1 *= d_scale_h*scale_h;
        faces[k].x2 *= d_scale_w*scale_w;
        faces[k].y2 *= d_scale_h*scale_h;

        for (int kk = 0; kk < 5; kk++) {
            faces[k].landmarks[2*kk]*= d_scale_w*scale_w;
            faces[k].landmarks[2*kk+1] *= d_scale_h*scale_h;
        }
    }
}

void dynamicScale(float in_w, float in_h)
{
    d_h = (int)(std::ceil(in_h / 32) * 32);
    d_w = (int)(std::ceil(in_w / 32) * 32);

    d_scale_h = in_h / d_h;
    d_scale_w = in_w / d_w;
}

void squareBox(std::vector<FaceInfo>& faces)
{
    float w = 0, h = 0, maxSize = 0;
    float cenx, ceny;
    for (int i = 0; i < faces.size(); i++) {
        w = faces[i].x2 - faces[i].x1;
        h = faces[i].y2 - faces[i].y1;

        maxSize = w < h ? h : w;
        cenx = faces[i].x1 + w / 2;
        ceny = faces[i].y1 + h / 2;

        faces[i].x1 =cenx - maxSize / 2 > 0 ? cenx - maxSize / 2 : 0;
        faces[i].y1 =ceny - maxSize / 2 > 0 ? ceny - maxSize / 2 : 0;
        faces[i].x2 = cenx + maxSize / 2 > image_w - 1 ? image_w - 1 : cenx + maxSize / 2;
        faces[i].y2 =ceny + maxSize / 2 > image_h-1 ? image_h - 1 : ceny + maxSize / 2;
    }
}

int detect(ncnn::Mat & inblob, std::vector<FaceInfo>& faces, int resized_w, int resized_h, float scoreThresh, float nmsThresh)
{
    if (inblob.empty()) {
    std::cout << "blob is empty ,please check!" << std::endl;
    return -1;
    }

    image_h = inblob.h;
    image_w = inblob.w;

    scale_w = (float)image_w / (float)resized_w;
    scale_h = (float)image_h / (float)resized_h;

    ncnn::Mat in;

    //scale
    dynamicScale(resized_w, resized_h);
    ncnn::resize_bilinear(inblob, in, d_w, d_h);

    ncnn::Extractor ex =net.create_extractor();
    __android_log_print(ANDROID_LOG_DEBUG, "centerface", "inblob size chw %d %d %d ",in.c,
            in.h,in.w);
    ex.input(centerface_param_id::BLOB_input_1, in); //"input.1"

    ncnn::Mat heatmap, scale, offset, landmarks;
    ex.extract(centerface_param_id::BLOB_537, heatmap); //537
    ex.extract(centerface_param_id::BLOB_538, scale); //538
    ex.extract(centerface_param_id::BLOB_539, offset); //539
    ex.extract(centerface_param_id::BLOB_540, landmarks); //540

    __android_log_print(ANDROID_LOG_DEBUG, "centerface", "heatmap size chw %d %d %d ",heatmap.c,heatmap.h,heatmap.w);
    decode(heatmap, scale,offset, landmarks,faces, scoreThresh,nmsThresh);
    squareBox(faces);
    return 0;
}

static struct timeval tv_begin;
static struct timeval tv_end;
static double elasped;

static void bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

static void bench_end(const char* comment)
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
//     fprintf(stderr, "%.2fms   %s\n", elasped, comment);
    __android_log_print(ANDROID_LOG_DEBUG, "WaterdemoNcnn", "%.2fms   %s", elasped, comment);
}

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Mat centerface_param;
static ncnn::Mat centerface_bin;
//static std::vector<std::string> squeezenet_words;
//static ncnn::Net centernet;

//static Centerface facenet;
//facenet





extern "C" {



JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "StyleTransferNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "StyleTransferNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

JNIEXPORT jboolean JNICALL Java_com_example_syj_centerface_centerface_Init(JNIEnv* env, jobject thiz, jbyteArray param, jbyteArray bin)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;



    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;


    net.opt = opt;

    // init param
    {
        int len = env->GetArrayLength(param);
        centerface_param.create(len, (size_t)1u);
        env->GetByteArrayRegion(param, 0, len, (jbyte*)centerface_param);
        int ret = net.load_param((const unsigned char*)centerface_param);
        __android_log_print(ANDROID_LOG_DEBUG, "waterNcnn", "load_param %d %d", ret, len);
    }

    // init bin
    {
        int len = env->GetArrayLength(bin);
        centerface_bin.create(len, (size_t)1u);
        env->GetByteArrayRegion(bin, 0, len, (jbyte*)centerface_bin);
        int ret = net.load_model((const unsigned char*)centerface_bin);
        __android_log_print(ANDROID_LOG_DEBUG, "waterNcnn", "load_model %d %d", ret, len);
    }

    // init words
//    {
//        int len = env->GetArrayLength(words);
//        std::string words_buffer;
//        words_buffer.resize(len);
//        env->GetByteArrayRegion(words, 0, len, (jbyte*)words_buffer.data());
//        squeezenet_words = split_string(words_buffer, "\n");
//    }

    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_example_syj_centerface_centerface_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return env->NewStringUTF("no vulkan capable gpu");
    }

    bench_start();


    std::string result_str = "success";
    ncnn::Mat inmat = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_RGB);

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int img_w = info.width;
    int img_h = info.height;
    cv::Mat image(img_h, img_w,CV_8UC3,cv::Scalar(0,1,2));
    inmat.to_pixels(image.data, ncnn::Mat::PIXEL_RGB); //8U3,0-255



//    ncnn::Mat inmat = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
    int over;
    over = detect(inmat, face_info, image.cols, image.rows, scoreThresh, nmsThresh);
    __android_log_print(ANDROID_LOG_DEBUG, "centerface", "num of face %d ",face_info.size());

    for (int i = 0; i < face_info.size(); i++) {
        __android_log_print(ANDROID_LOG_DEBUG, "centerface", "xyxy %f %f %f %f ",face_info[i].x1,
                            face_info[i].y1,face_info[i].x2,face_info[i].y2);
        cv::rectangle(image, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);
        for (int j = 0; j < 5; j++) {
            cv::circle(image, cv::Point(face_info[i].landmarks[2*j], face_info[i].landmarks[2*j+1]), 2, cv::Scalar(255, 255, 0), 2);
        }
    }

    ncnn::Mat out;
    out = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows);
    out.to_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_RGB);





    // +10 to skip leading n03179701
    jstring result = env->NewStringUTF(result_str.c_str());

    bench_end("detect");

    return result;
}

}
