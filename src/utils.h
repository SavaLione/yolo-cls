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
 * @file utils.h
 * @brief Declares utility functions for argument parsing, string conversion, file type checking, and the functions that run in separate threads.
 * @author Savelii Pototskii
 * @date 2025-08-17
 * @copyright Copyright (C) 2025 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
*/
#ifndef UTILS_H
#define UTILS_H

#include "tsqueue.h"
#include "yolo.h"

#include <thread>

/**
 * @brief Converts a string with a storage unit (e.g., `100mb`, `2g`) to a numeric value in bytes.
 * @param[in] unit The string representation of the size (e.g., `100mb`). Case-insensitive.
 * @return The equivalent number of bytes as a uint64_t.
 * @throws std::invalid_argument if the string format is invalid.
 * @throws std::overflow_error if the resulting value is too large for uint64_t.
 */
uint64_t string_unit_to_numeric(std::string const &unit);

/**
 * @brief Checks if a file extension corresponds to an image format supported by OpenCV.
 * @param[in] extension The file extension (e.g., `.jpg`, `png`). Case-insensitive.
 * @return True if the extension is supported, false otherwise.
 */
bool is_supported_image(std::string_view extension);

/**
 * @struct configuration
 * @brief Holds the application's configuration settings, parsed from command-line arguments.
 */
struct configuration
{
    std::string model_path       = "";                                  ///< Path to the ONNX model file.
    std::string classes_path     = "";                                  ///< Path to the text file with class names.
    int top_k                    = 5;                                   ///< Number of top classification results to show.
    unsigned int threads         = std::thread::hardware_concurrency(); ///< Number of worker threads.
    bool enable_timing           = false;                               ///< If true, include processing time in the output.
    bool use_softmax             = false;                               ///< If true, apply softmax to model output.
    uint64_t max_filesize        = string_unit_to_numeric("100mb");     ///< Maximum allowed image file size in bytes.
    bool disable_extension_check = false;                               ///< If true, do not check file extensions.
    std::vector<std::string> image_files;                               ///< List of image files from command-line arguments.
};

/**
 * @brief Parses command-line arguments and populates a configuration struct.
 * @param argc Argument count from `main`.
 * @param argv Argument vector from `main`.
 * @return A populated `configuration` struct.
 * @throws std::runtime_error on parsing failure or invalid arguments.
 */
configuration parse_arguments(int argc, char **argv);

/**
 * @brief The main worker thread function.
 *        Pops a file path from the input queue, performs classification,
 *        formats the result, and pushes it to the output queue.
 * @param tsq_in The thread-safe input queue for file paths.
 * @param tsq_out The thread-safe output queue for formatted results.
 * @param model The YOLO model instance to use for classification.
 * @param[in] c The application configuration.
 */
void thread_classify(tsqueue &tsq_in, tsqueue &tsq_out, yolo &model, configuration const &c);

/**
 * @brief The output thread function.
 *        Pops formatted results from the output queue and prints them to standard output.
 * @param tsq The thread-safe output queue.
 */
void thread_print_tsq(tsqueue &tsq);

/**
 * @brief The input thread function for piped data.
 *        Reads lines (file paths) from standard input and pushes them to the input queue.
 * @param tsq_in The thread-safe input queue to push file paths to.
 * @param[in] c The application configuration (used for extension checking).
 */
void thread_get_line(tsqueue &tsq_in, configuration const &c);

/**
 * @brief Prints help information that is invoked by `-h` or `--help`
*/
void print_help();

/**
 * @brief Prints about information that is invoked by `-a` or `--about`
*/
void print_about();

#endif // UTILS_H