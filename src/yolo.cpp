/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Copyright (C) 2025 Savelii Pototskii (savalione.com)
 *
 * Author: Savelii Pototskii <savelii.pototskii@gmail.com>
 *
 * This file is part of yolo-cls.
 *
 * yolo-cls is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * yolo-cls is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with yolo-cls. If not, see <https://www.gnu.org/licenses/>.
*/
/**
 * @file yolo.cpp
 * @brief Defines interfaces for YOLO classification inference.
 * @author Savelii Pototskii
 * @date 2025-08-10
 * @copyright Copyright (C) 2025 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
*/
#include "yolo.h"

#include <fstream>
#include <filesystem>
#include <string>

/**
 * @brief Default constructor.
 * @warning  It is in non-predicting state. The session is nullptr, and other members are default-initialized.
 */
yolo::yolo() : env(ORT_LOGGING_LEVEL_WARNING, "yolo-cls-uninitialized"), session(nullptr)
{
    // It is in non-predicting state.
    // The session is nullptr, and other members are default-initialized.
}

/**
 * @brief Move constructor.
 * @param[in] other The yolo object to move from.
 */
yolo::yolo(yolo &&other) noexcept
    : env(std::move(other.env)),
      session(std::move(other.session)),
      allocator(std::move(other.allocator)),
      input_width(other.input_width),
      input_height(other.input_height),
      input_node_names(std::move(other.input_node_names)),
      output_node_names(std::move(other.output_node_names)),
      input_names(std::move(other.input_names)),
      output_names(std::move(other.output_names)),
      class_names(std::move(other.class_names)),
      input_nodes_num(other.input_nodes_num),
      output_nodes_num(other.output_nodes_num),
      use_softmax(other.use_softmax)
{
    // The `other` object is now in a moved-from state (the same as a default-constructed object)
    other.input_width      = 0;
    other.input_height     = 0;
    other.input_nodes_num  = 0;
    other.output_nodes_num = 0;
    other.use_softmax      = false;
}

/**
 * @brief Move assignment operator.
 * @param[in] other The yolo object to move from.
 * @return A reference to this object.
 */
yolo &yolo::operator=(yolo &&other) noexcept
{
    if(this != &other)
    {
        env               = std::move(other.env);
        session           = std::move(other.session);
        allocator         = std::move(other.allocator);
        input_width       = other.input_width;
        input_height      = other.input_height;
        input_node_names  = std::move(other.input_node_names);
        output_node_names = std::move(other.output_node_names);
        input_names       = std::move(other.input_names);
        output_names      = std::move(other.output_names);
        class_names       = std::move(other.class_names);
        input_nodes_num   = other.input_nodes_num;
        output_nodes_num  = other.output_nodes_num;
        use_softmax       = other.use_softmax;

        // Reset the `other` object
        other.input_width      = 0;
        other.input_height     = 0;
        other.input_nodes_num  = 0;
        other.output_nodes_num = 0;
        other.use_softmax      = false;
    }
    return *this;
}

/**
 * @brief Constructs and initializes a yolo object with an option to enable softmax.
 * @param[in] model_path Path to the ONNX model file.
 * @param[in] cls_path Path to the text file containing class names (one per line).
 * @param[in] use_softmax If true, applies softmax to the model's output scores to convert them to probabilities.
 * @throws std::invalid_argument if the model or class file cannot be loaded or is invalid.
 * @throws std::filesystem::filesystem_error if the class names file cannot be opened.
 */
yolo::yolo(std::string const &model_path, std::string const &cls_path, bool const &use_softmax) : yolo(model_path, cls_path)
{
    this->use_softmax = use_softmax;
}

/**
 * @brief Constructs and initializes a yolo object by loading the ONNX model and class names.
 * @param[in] model_path Path to the ONNX model file.
 * @param[in] cls_path Path to the text file containing class names (one per line).
 * @throws std::invalid_argument if the model or class file cannot be loaded or is invalid.
 * @throws std::filesystem::filesystem_error if the model or class names file cannot be opened, read, or does not exist.
 */
yolo::yolo(std::string const &model_path, std::string const &cls_path) : env(ORT_LOGGING_LEVEL_WARNING, "yolo-cls")
{
    // Read the model file into a memory buffer
    std::ifstream model_stream(model_path, std::ios::binary | std::ios::ate);
    if(!model_stream.is_open())
        throw std::filesystem::filesystem_error("Could not open model file", model_path, std::make_error_code(std::errc::io_error));

    std::streamsize model_size = model_stream.tellg();
    model_stream.seekg(0, std::ios::beg);

    std::vector<char> model_buffer(model_size);
    if(!model_stream.read(model_buffer.data(), model_size))
        throw std::filesystem::filesystem_error("Could not read model file", model_path, std::make_error_code(std::errc::io_error));

    model_stream.close();

    // Create ONNX runtime session from the memory buffer
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session = Ort::Session(env, model_buffer.data(), model_buffer.size(), session_options);

    input_nodes_num  = session.GetInputCount();
    output_nodes_num = session.GetOutputCount();

    if(input_nodes_num == 0)
        throw std::invalid_argument("Model file '" + model_path + "' has no input nodes.");

    if(output_nodes_num == 0)
        throw std::invalid_argument("Model file '" + model_path + "' has no output nodes.");

    // Get input/output node details
    input_node_names.push_back(session.GetInputNameAllocated(0, allocator));
    output_node_names.push_back(session.GetOutputNameAllocated(0, allocator));

    input_names.push_back(input_node_names.back().get());
    output_names.push_back(output_node_names.back().get());

    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto tensor_info              = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_dims               = tensor_info.GetShape(); // Shape is [batch, channels, height, width]

    input_height = input_dims[2];
    input_width  = input_dims[3];

    // Load class names from file
    auto const &path = cls_path;

    if(!std::filesystem::is_regular_file(path))
        throw std::filesystem::filesystem_error("Class names path is not a regular file or does not exist", path, std::make_error_code(std::errc::no_such_file_or_directory));

    std::ifstream ifs(path);
    if(!ifs.is_open())
        throw std::filesystem::filesystem_error("Could not open class names file", path, std::make_error_code(std::errc::io_error));

    std::string line;
    while(std::getline(ifs, line))
    {
        class_names.push_back(line);
    }

    ifs.close();
}

/*
// Different implementations of `preprocess`.
// Can be useful if OpenCV version is < 4.6.0 (due to `cv::dnn::blobFromImage(resized_image, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);`)

void yolo::preprocess(const cv::Mat &image, std::vector<float> &output_tensor)
{
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));

    // Convert BGR to RGB
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    // Convert image from uint8 [0, 255] to float [0, 1]
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

    // Reshape image from HWC to NCHW
    //
    // HWC:
    //   H - Height
    //   W - Width
    //   C - Channels
    //
    // NCHW:
    //   B - Batch
    //   C - Channels
    //   H - Height
    //   W - Width
    //
    // N is 1 (batch size)
    // C is 3 (RGB)
    output_tensor.resize(1 * 3 * input_height * input_width);
    float *output_ptr = output_tensor.data();

    for(int c = 0; c < 3; ++c)
    {
        for(int h = 0; h < input_height; ++h)
        {
            for(int w = 0; w < input_width; ++w)
            {
                output_ptr[c * (input_height * input_width) + h * input_width + w] = resized_image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}

void yolo::preprocess(const cv::Mat &image, std::vector<float> &output_tensor)
{
    cv::Mat resized_image;
    cv::Mat res_img;
    // cv::resize(image, resized_image, cv::Size(input_width, input_height));

    // Convert BGR to RGB
    cv::cvtColor(image, resized_image, cv::COLOR_BGR2RGB);

    resized_image = cv::dnn::blobFromImage(resized_image, 1.0, cv::Size(input_width, input_height), cv::Scalar(), false, true, CV_32F);

    // Convert image from uint8 [0, 255] to float [0, 1]
    resized_image.convertTo(res_img, CV_32F, 1.0 / 255.0);

    // Reshape to a single row
    res_img = res_img.reshape(1, 1);

    output_tensor.assign((float *)res_img.ptr(0), (float *)res_img.ptr(0) + res_img.total() * res_img.channels());
}
*/

void yolo::preprocess(cv::Mat const &image, std::vector<float> &output_tensor) const
{
    cv::Mat resized_image;

    cv::resize(image, resized_image, cv::Size(input_width, input_height));

    // Convert BGR to RGB
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    resized_image = cv::dnn::blobFromImage(resized_image, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);

    // Convert image from uint8 [0, 255] to float [0, 1]
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

    // Reshape to a single row
    resized_image = resized_image.reshape(1, 1);

    output_tensor.assign((float *)resized_image.ptr(0), (float *)resized_image.ptr(0) + resized_image.total() * resized_image.channels());
}

void yolo::softmax(std::vector<float> &scores) const
{
    if(scores.empty())
        return;

    float max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<float> exp_scores;
    exp_scores.reserve(scores.size());

    float sum_exp_scores = 0.0f;
    for(float s : scores)
    {
        float exp_s = std::exp(s - max_score);
        exp_scores.push_back(exp_s);
        sum_exp_scores += exp_s;
    }

    for(size_t i = 0; i < scores.size(); ++i)
    {
        scores[i] = exp_scores[i] / sum_exp_scores;
    }
}

/**
 * @brief Performs classification on a given image.
 * @param[in] image The input image as a `cv::Mat` object.
 * @param[in] top_k The number of top predictions to return.
 * @return A vector of `prediction` structs, sorted by confidence in descending order.
 * @throws std::runtime_error if the model is not initialized (e.g., default-constructed).
 */
std::vector<prediction> yolo::predict(cv::Mat const &image, size_t const &top_k)
{
    // Check if the model is initialized
    if(session == nullptr)
        throw std::runtime_error("The model is not initialized.");

    // Pre-process the image
    std::vector<float> input_tensor_values;
    preprocess(image, input_tensor_values);

    // Create input tensor object
    std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
    auto memory_info                 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor          = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Run inference
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions {nullptr}, input_names.data(), &input_tensor, input_nodes_num, output_names.data(), output_nodes_num);

    // Post-process the output
    float *raw_output  = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape  = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = output_shape[1]; // Number of classes

    std::vector<float> scores(raw_output, raw_output + output_size);

    // Apply softmax to get probabilities
    if(use_softmax)
        softmax(scores);

    // Create a vector of pairs (index, score) to keep track of original indices
    std::vector<std::pair<int, float>> indexed_scores;
    indexed_scores.reserve(scores.size());
    for(int i = 0; i < scores.size(); ++i)
    {
        indexed_scores.emplace_back(i, scores[i]);
    }

    // Sort the vector of pairs in descending order based on the score
    std::sort(indexed_scores.begin(), indexed_scores.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

    // Get the top K results
    std::vector<prediction> top_predictions;
    size_t count = std::min(top_k, indexed_scores.size());
    for(size_t i = 0; i < count; ++i)
    {
        int class_index  = indexed_scores[i].first;
        float confidence = indexed_scores[i].second;

        if(class_index <= class_names.size())
            top_predictions.push_back({class_names[class_index], confidence});
        else
            top_predictions.push_back({"class_" + std::to_string(class_index), confidence});
    }

    return top_predictions;
}
