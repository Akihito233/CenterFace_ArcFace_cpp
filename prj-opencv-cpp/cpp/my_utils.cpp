#include "my_utils.h"


cv::Mat align(cv::Mat& image, FaceInfo& face) {

	float scale = std::min(112.0f / (face.x2-face.x1), 112.0f / (face.y2-face.y1));
	std::vector<cv::Point2f> points_from;

	cv::Size2f resizeSizeTemp = {image.size().width * scale, image.size().height * scale};
	cv::Mat image_clone = image.clone();
	cv::resize(image_clone, image_clone, resizeSizeTemp);

	for (int i = 0; i < 5; i++) {
		points_from.emplace_back(face.landmarks[2 * i] * scale, face.landmarks[2 * i + 1] * scale);
		// std::cout<<"landmarks: "<<face.landmarks[2 * i] * scale<<", "<<face.landmarks[2 * i + 1] * scale<<std::endl;

		cv::circle(image_clone, points_from[i], 2, cv::Scalar(255, 255, 0), 2);
		
	}

	// cv::InputArray inputArray_from(points_from);
	std::vector<cv::Point2f> points_to = { {37.5f, 51.5f},
		{74.5f, 51.5f},
		{56.0f, 71.74f},
		{40.5f, 92.25f},
		{71.5f, 92.25f} };
	// cv::InputArray inputArray_to(points_to);

	for (int j = 0; j < 5; j++) {
		cv::circle(image_clone, points_to[j], 2, cv::Scalar(0, 0, 255), 2);
	}
	// cv::imwrite("../data/results/image_clone.jpg", image_clone);

	// float scaleX = cv::norm(points_to[1] - points_to[0]) / cv::norm(points_from[1] - points_from[0]);
    	// float scaleY = cv::norm(points_to[3] - points_to[2]) / cv::norm(points_from[3] - points_from[2]);
    	// points_to[2].x /= scaleX;
    	// points_to[2].y /= scaleY;
	// for (auto& point : points_to) {
        	// point.x /= scaleX;
        	// point.y /= scaleY;
    	// }

	cv::Mat M = cv::estimateAffinePartial2D(points_from, points_to);

	// std::cout<<"Matrix: "<<M<<std::endl;

	cv::Mat resizedImage;
	cv::Mat warpedImage;
	cv::Size2f resizeSize = {image.size().width * scale, image.size().height * scale};
	// size of the output image
	cv::Size size = { 112, 112 };
	cv::resize(image, resizedImage, resizeSize);
	cv::warpAffine(resizedImage, warpedImage, M, size);

	return warpedImage;
}

void prepareFaceData(std::string& face_data_path, lite::onnxruntime::cv::faceid::GlintArcFace* glint_arcface, std::vector<lite::types::FaceContent>& faces_data, std::vector<std::string>& names) {
	//names.emplace_back("Unknown");
	std::filesystem::path folderPath(face_data_path);
	if (std::filesystem::exists(folderPath) && std::filesystem::is_directory(folderPath)) {
		for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
			if (entry.is_directory()) {
				std::cout << "directory: " << entry.path().string() << std::endl;
				names.emplace_back(entry.path().filename());

				for (const auto& imageEntry : std::filesystem::directory_iterator(entry.path())) {
					if (imageEntry.is_regular_file()) {
						if (imageEntry.path().extension() == ".jpg" || imageEntry.path().extension() == ".png") {
							cv::Mat image = cv::imread(imageEntry.path().string());

							if (image.empty()) {
								std::cout << "Unable to read face_ Data image file: " << imageEntry.path().string() << std::endl;
							}
							else {
								lite::types::FaceContent face_content_temp;
								glint_arcface->detect(image, face_content_temp);
								if (face_content_temp.flag) {
									faces_data.emplace_back(face_content_temp);
								}
								else {
									std::cout << "face_content.flag == false: " << imageEntry.path().string() << std::endl;
								}
								//cv::imshow("Image", image);
								//cv::waitKey(0);
							}
						}
					}
				}

			}
		}
	}
	else {
		std::cout << "The folder does not exist or the path is incorrect!" << std::endl;
	}

	std::ofstream facesFile(folderPath / "faces_data.bin", std::ios::binary);
	if (facesFile.is_open()) {
		for (const auto& face : faces_data) {
			facesFile.write(reinterpret_cast<const char*>(&face.dim), sizeof(unsigned int));
			facesFile.write(reinterpret_cast<const char*>(face.embedding.data()), face.dim * sizeof(float));
			facesFile.write(reinterpret_cast<const char*>(&face.flag), sizeof(bool));
		}
		facesFile.close();
	}

	std::ofstream namesFile(folderPath / "names.bin", std::ios::binary);
	if (namesFile.is_open()) {
		for (const auto& name : names) {
			size_t length = name.size();
			namesFile.write(reinterpret_cast<const char*>(&length), sizeof(size_t));
			namesFile.write(reinterpret_cast<const char*>(name.data()), length);
		}
		namesFile.close();
	}
}

void loadFacesData(std::string& face_data_path, std::vector<lite::types::FaceContent>& faces_data, std::vector<std::string>& names)
{
	std::filesystem::path folderPath(face_data_path);
	std::ifstream facesFile(folderPath / "faces_data.bin", std::ios::binary);
	if (facesFile.is_open()) {
		while (true) {
			unsigned int dim;
			facesFile.read(reinterpret_cast<char*>(&dim), sizeof(unsigned int));
			if (facesFile.eof())
				break;
			std::vector<float> embedding(dim);
			facesFile.read(reinterpret_cast<char*>(embedding.data()), dim * sizeof(float));
			bool flag;
			facesFile.read(reinterpret_cast<char*>(&flag), sizeof(bool));
			faces_data.emplace_back(lite::types::FaceContent( embedding, dim, flag ));
		}
		facesFile.close();
	}

	std::ifstream namesFile(folderPath / "names.bin", std::ios::binary);
	if (namesFile.is_open()) {
		while (true) {
			size_t length;
			namesFile.read(reinterpret_cast<char*>(&length), sizeof(size_t));
			if (namesFile.eof())
				break;
			std::string name(length, '\0');
			namesFile.read(reinterpret_cast<char*>(name.data()), length);
			names.emplace_back(name);
		}
		namesFile.close();
	}
}

void infer(std::vector<cv::Mat>& face_images, std::vector<lite::types::FaceContent>& faces_data, lite::onnxruntime::cv::faceid::GlintArcFace* glint_arcface, std::vector<int>& face_idx, std::vector<float>& similarity, const float& threshold) {
	for (auto& face_image : face_images) {
		lite::types::FaceContent face_content;
		glint_arcface->detect(face_image, face_content);
		int idx = -1;
		float max_sim = -1.0f;
		if (face_content.flag) {
			for (int i = 0; i < faces_data.size(); i++) {
				float sim = lite::utils::math::cosine_similarity<float>(
					face_content.embedding, faces_data[i].embedding);
				if (sim >= threshold) {
					if (idx == -1 || sim > max_sim) {
						idx = i;
						max_sim = sim;
					}
				}
			}
		}
		face_idx.emplace_back(idx);
		similarity.emplace_back(max_sim);
	}
}
