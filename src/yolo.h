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
 * @file yolo.h
 * @brief Defines interfaces for YOLO classification inference.
 * @author Savelii Pototskii
 * @date 2025-08-10
 * @copyright Copyright (C) 2025 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
*/
#ifndef YOLO_H
#define YOLO_H

#include <cstddef>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

/**
 * @struct prediction
 * @brief A structure to hold a single classification prediction.
 */
struct prediction
{
    std::string class_name; ///< The name of the predicted class.
    float confidence;       ///< The confidence score of the prediction.
};

/**
 * @class yolo
 * @brief Encapsulates the YOLO classification model, handling model loading, preprocessing, inference, and post-processing.
 */
class yolo
{
public:
    /**
     * @brief Default constructor.
     * @warning  It is in non-predicting state. The session is nullptr, and other members are default-initialized.
     */
    yolo();

    /**
     * @brief Constructs and initializes a yolo object by loading the ONNX model and class names.
     * @param[in] model_path Path to the ONNX model file.
     * @param[in] cls_path Path to the text file containing class names (one per line).
     * @throws std::invalid_argument if the model or class file cannot be loaded or is invalid.
     * @throws std::filesystem::filesystem_error if the class names file cannot be opened.
     */
    yolo(std::string const &model_path, std::string const &cls_path);

    /**
     * @brief Constructs and initializes a yolo object with an option to enable softmax.
     * @param[in] model_path Path to the ONNX model file.
     * @param[in] cls_path Path to the text file containing class names (one per line).
     * @param[in] use_softmax If true, applies softmax to the model's output scores to convert them to probabilities.
     * @throws std::invalid_argument if the model or class file cannot be loaded or is invalid.
     * @throws std::filesystem::filesystem_error if the class names file cannot be opened.
     */
    yolo(std::string const &model_path, std::string const &cls_path, bool const &use_softmax);

    // Rule of five - disable copying, enable moving
    yolo(const yolo &)            = delete;
    yolo &operator=(const yolo &) = delete;

    /**
     * @brief Move constructor.
     * @param[in] other The yolo object to move from.
     */
    yolo(yolo &&other) noexcept;

    /**
     * @brief Move assignment operator.
     * @param[in] other The yolo object to move from.
     * @return A reference to this object.
     */
    yolo &operator=(yolo &&other) noexcept;

    /**
     * @brief Performs classification on a given image.
     * @param[in] image The input image as a `cv::Mat` object.
     * @param[in] top_k The number of top predictions to return.
     * @return A vector of `prediction` structs, sorted by confidence in descending order.
     * @throws std::runtime_error if the model is not initialized (e.g., default-constructed).
     */
    std::vector<prediction> predict(cv::Mat const &image, size_t const &top_k);

private:
    // ONNX Runtime session members
    Ort::Env env;
    Ort::Session session {nullptr};
    Ort::AllocatorWithDefaultOptions allocator;

    // Model properties extracted from the ONNX file
    int64_t input_width  = 0;
    int64_t input_height = 0;
    std::vector<Ort::AllocatedStringPtr> input_node_names;
    std::vector<Ort::AllocatedStringPtr> output_node_names;

    // Pointers for the ONNX Runtime API
    std::vector<char const *> input_names;
    std::vector<char const *> output_names;

    // Class names loaded from the provided text file
    std::vector<std::string> class_names;

    /**
     * @brief Prepares an image for inference.
              This involves resizing, color space conversion (BGR to RGB),
              normalization (to [0, 1]), and layout conversion to NCHW format.
     * @param[in] image The input image.
     * @param[out] output_tensor A vector to be filled with the preprocessed image data.
     */
    void preprocess(cv::Mat const &image, std::vector<float> &output_tensor) const;

    /**
     * @brief Applies the softmax function to a vector of raw scores (logits) to convert them into probabilities.
     * @param[out] scores A vector of scores to be modified in-place.
     */
    void softmax(std::vector<float> &scores) const;

    // Model input/output node counts
    size_t input_nodes_num  = 0;
    size_t output_nodes_num = 0;

    /// Flag indicating whether to apply softmax to the output scores.
    bool use_softmax = false;
};

#endif // YOLO_H