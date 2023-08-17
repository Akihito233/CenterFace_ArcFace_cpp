#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "cv_dnn_centerface.h"
#include "lite/lite.h"
#include "my_utils.h"

static void test_onnxruntime();

int main(int argc, char** argv) {
	// test_onnxruntime();

	if (argc != 3) {
		std::cout << " ./demo(.exe) video_file output_filename " << std::endl;
		return -1;
	}

	std::string centerface_model_path = "../../models/onnx/centerface.onnx";
	std::string video_file = argv[1];
	std::string output_filename = argv[2];
	std::string output_filename_prefix = "../data/results/";
	std::string recognition_model_path = "../../models/onnx/glint360k_r50.onnx";
	float threshold = 0.5f;
	std::string face_data_path = "../data/face_data";
	std::vector<lite::types::FaceContent> faces_data;
	std::vector<std::string> names;

	Centerface centerface(centerface_model_path,640,480);
	lite::onnxruntime::cv::faceid::GlintArcFace *glint_arcface =
		new lite::onnxruntime::cv::faceid::GlintArcFace(recognition_model_path);

	prepareFaceData(face_data_path, glint_arcface, faces_data, names);
	std::cout<<"prepare: ok"<<std::endl;
	
	cv::VideoCapture cap(video_file);
	if(!cap.isOpened()) {
		std::cout << "Failed to open video file." << std::endl;
		return -1;
	}
	int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(cv::CAP_PROP_FPS);
	std::string output_video_filename = output_filename_prefix + output_filename + ".mp4";
	std::string output_txt_filename = output_filename_prefix + output_filename + ".txt";
	cv::VideoWriter output(output_video_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));
	std::ofstream output_txt(output_txt_filename, std::ios::out | std::ios::trunc);
	if(!output.isOpened()) {
		std::cout << "Failed to create output video file." << std::endl;
		return -1;
	}
	if(!output_txt.is_open()) {
                std::cout << "Failed to open output txt file." << std::endl;
                return -1;
        }
	unsigned int cnt = 0;
	cv::Mat frame;
	while(cap.read(frame)) {
		std::vector<FaceInfo> face_info;
		cnt += 1;

		output_txt << "image " << cnt << " :\n";
	
		centerface.detect(frame, face_info);
		// std::cout<<"detect: ok"<<std::endl;

		if(face_info.size() == 0) {
			output.write(frame);
			output_txt << "no face" << std::endl;
			continue;
		}
		else {
			std::vector<cv::Mat> face_images;
			for (int i = 0; i < face_info.size(); i++) {
				cv::Mat image_raw = frame.clone();
				cv::rectangle(frame, cv::Point2f(face_info[i].x1, face_info[i].y1), cv::Point2f(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);
				for (int j = 0; j < 5; j++) {
					cv::circle(frame, cv::Point2f(face_info[i].landmarks[2*j], face_info[i].landmarks[2*j+1]), 2, cv::Scalar(255, 255, 0), 2);
				}
				face_images.emplace_back(align(image_raw, face_info[i]));
				// std::cout<<"align: ok"<<std::endl;
				// cv::imwrite("../data/results/align_"+std::to_string(cnt)+".jpg", face_images[i]);
			}
			std::vector<int> face_idx;
			std::vector<float> similarity;
			infer(face_images, faces_data, glint_arcface, face_idx, similarity, threshold);
			// std::cout<<"infer: ok"<<std::endl;
			// std::cout<<"face_images_size: "<<face_images.size()<<std::endl;
			// std::cout<<"face_idx_size: "<<face_idx.size()<<std::endl;
			// std::cout<<"face_similarity_size: "<<similarity.size()<<std::endl;
			// std::cout<<"names_size: "<<names.size()<<std::endl;
			for (int j = 0; j < face_images.size(); j++) {
				std::string str;
				
				// std::cout<<"face_idx: "<<face_idx[j]<<std::endl;

				if(face_idx[j] == -1) str = "Unknown";
				else str = names[face_idx[j]] + ": " + std::to_string(similarity[j]);
				// std::cout << "identity : " + names[face_idx[j]] + ", cosine similarity : " + std::to_string(similarity[j]) + "\n";
				std::cout<<str<<std::endl;

				// drawText(face_info[j], str, frame);
				cv::putText(frame, str, cv::Point2f(face_info[j].x1, face_info[j].y1), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
				output_txt << "box coordinates : top-left : (" + std::to_string(face_info[j].x1) + ", " + std::to_string(face_info[j].y1) +
				       	"), bottom-right : (" + std::to_string(face_info[j].x2) + ", " + std::to_string(face_info[j].y2) + ")\n";
				// output_txt << "identity : " + names[face_idx[j]] + ", cosine similarity : " + std::to_string(similarity[j]) + "\n";
				output_txt << str << std::endl;
			}

			// cv::imshow("test", frame);
			output.write(frame);
			// if (cv::waitKey(0) == 'q') {
				// break;
			// }
		}
	}

	cap.release();
	output.release();
	output_txt.close();
	cv::destroyAllWindows();

	return 0;
}

