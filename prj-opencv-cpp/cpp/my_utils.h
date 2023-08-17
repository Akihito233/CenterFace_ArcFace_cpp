#ifndef MY_UTILS_H
#define MY_UTILS_H

#include "cv_dnn_centerface.h"
#include "lite/lite.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

cv::Mat align(cv::Mat& image, FaceInfo& face);
void prepareFaceData(std::string& face_data_path, lite::onnxruntime::cv::faceid::GlintArcFace* glint_arcface, std::vector<lite::types::FaceContent>& faces_data, std::vector<std::string>& names);
void loadFacesData(std::string& face_data_path, std::vector<lite::types::FaceContent>& faces_data, std::vector<std::string>& names);
void infer(std::vector<cv::Mat>& face_images, std::vector<lite::types::FaceContent>& faces_data, lite::onnxruntime::cv::faceid::GlintArcFace* glint_arcface, std::vector<int>& face_idx, std::vector<float>& similarity, const float& threshold);

#endif
