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
 * @file utils.cpp
 * @brief Declares utility functions for argument parsing, string conversion, file type checking, and the functions that run in separate threads.
 * @author Savelii Pototskii
 * @date 2025-08-17
 * @copyright Copyright (C) 2025 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
*/
#include "utils.h"

#include <algorithm>
#include <stdexcept>
#include <map>
#include <limits>
#include <string>
#include <unordered_set>
#include <filesystem>

#include "xgetopt/xgetopt.h"
#include "config.h"

/**
 * @brief Converts a string with a storage unit (e.g., `100mb`, `2g`) to a numeric value in bytes.
 * @param[in] unit The string representation of the size (e.g., `100mb`). Case-insensitive.
 * @return The equivalent number of bytes as a uint64_t.
 * @throws std::invalid_argument if the string format is invalid.
 * @throws std::overflow_error if the resulting value is too large for uint64_t.
 */
uint64_t string_unit_to_numeric(std::string const &unit)
{
    std::string s = unit;

    // Trim leading whitespace
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));

    // Trim trailing whitespace
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());

    if(s.empty())
        throw std::invalid_argument("Input string for unit conversion cannot be empty.");

    // Convert string to lowercase
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });

    // Find the split between the number and the unit
    size_t split_position = s.find_first_not_of("0123456789");

    std::string number_str = s.substr(0, split_position);
    std::string unit_str   = (split_position == std::string::npos) ? "" : s.substr(split_position);

    if(number_str.empty())
        throw std::invalid_argument("Input string '" + unit + "' does not contain a numeric part.");

    // Convert string to number
    uint64_t number = std::stoull(number_str);

    // Determine the multiplier based on the unit
    uint64_t multiplier = 1;

    static const std::map<std::string, uint64_t> multipliers = {
        {"b", 1ULL},
        {"k", 1024ULL},
        {"kb", 1024ULL},
        {"m", 1024ULL * 1024},
        {"mb", 1024ULL * 1024},
        {"g", 1024ULL * 1024 * 1024},
        {"gb", 1024ULL * 1024 * 1024},
        {"t", 1024ULL * 1024 * 1024 * 1024},
        {"tb", 1024ULL * 1024 * 1024 * 1024},
    };

    auto it = multipliers.find(unit_str);
    if(it != multipliers.end())
    {
        multiplier = it->second;
    }
    else if(unit_str.empty())
    {
        multiplier = 1;
    }
    else
    {
        throw std::invalid_argument("Unknown storage unit '" + unit_str + "' in input '" + unit + "'.");
    }

    // Overflow check. Prevent wraparound
    const uint64_t max_val = std::numeric_limits<uint64_t>::max();

    // Overflow check
    if(multiplier > 1 && number > max_val / multiplier)
        throw std::overflow_error("The value '" + unit + "' results in an overflow and is too large to be represented.");

    return number * multiplier;
}

/**
 * @brief Checks if a file extension corresponds to an image format supported by OpenCV.
 * @param[in] extension The file extension (e.g., `.jpg`, `png`). Case-insensitive.
 * @return True if the extension is supported, false otherwise.
 */
bool is_supported_image(std::string_view extension)
{
    /*
        OpenCV supported image formats:
            Windows bitmaps - *.bmp, *.dib (always supported)
            JPEG files - *.jpeg, *.jpg, *.jpe (see the Note section)
            JPEG 2000 files - *.jp2 (see the Note section)
            Portable Network Graphics - *.png (see the Note section)
            WebP - *.webp (see the Note section)
            Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm (always supported)
            Sun rasters - *.sr, *.ras (always supported)
            TIFF files - *.tiff, *.tif (see the Note section)
            OpenEXR Image files - *.exr (see the Note section)
            Radiance HDR - *.hdr, *.pic (always supported)
            Raster and Vector geospatial data supported by GDAL (see the Note section)
        GDAL Image Formats:
            DTED: Military Elevation Data (.dt0, .dt1, .dt2)
            EHdr: ESRI .hdr Labelled:
            ENVI: ENVI .hdr Labelled Raster
            HFA: Erdas Imagine (.img)
            JP2MrSID: JPEG2000 (.jp2, .j2k)
            MrSID: Multi-resolution Seamless Image Database
            NITF: National Imagery Transmission Format
            ECW: ERDAS Compressed Wavelets (.ecw):
            JP2ECW: JPEG2000 (.jp2, .j2k)
            AIG: Arc/Info Binary Grid
            JP2KAK: JPEG2000 (.jp2, .j2k)
        
        Check these links for more information:
        * https://docs.opencv.org/4.6.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
        * https://docs.geoserver.org/main/en/user/data/raster/gdal.html
    */

    // clang-format off
    static const std::unordered_set<std::string> supported_extensions = {
        // OpenCV
        "bmp",  "dib",
        "jpeg", "jpg",
        "jpe",  "jp2",
        "png",  "webp",
        "pbm",  "pgm",
        "ppm",  "pxm",
        "pnm",  "sr",
        "ras",  "tiff",
        "tif",  "exr",
        "hdr",  "pic",

        // GDAL
        "dt0", "dt1",
        "dt2", "hdr",
        "img", "j2k",
        "ecw",
    };
    // clang-format on

    // Remove the leading dot, if it exists
    if(!extension.empty() && extension.front() == '.')
        extension.remove_prefix(1);

    // Convert string to lowercase
    std::string lower_extension;
    lower_extension.reserve(extension.size());
    std::transform(extension.begin(), extension.end(), std::back_inserter(lower_extension), [](unsigned char c) { return std::tolower(c); });

    return supported_extensions.count(lower_extension) > 0;
}

/**
 * @brief Parses command-line arguments and populates a configuration struct.
 * @param argc Argument count from `main`.
 * @param argv Argument vector from `main`.
 * @return A populated `configuration` struct.
 * @throws std::runtime_error on parsing failure or invalid arguments.
 */
configuration parse_arguments(int argc, char **argv)
{
    if(argc == 1)
    {
        print_help();
        exit(EXIT_SUCCESS);
    }

    configuration result;

    // Accepted parameters
    std::string const short_opts = "m:c:k:t:TSF:Dhva";

    // clang-format off
    std::array<xoption, 12> long_options =
        {{
            {"model",               xrequired_argument, nullptr, 'm'},
            {"classes",             xrequired_argument, nullptr, 'c'},
            {"top-k",               xrequired_argument, nullptr, 'k'},
            {"threads",             xrequired_argument, nullptr, 't'},
            {"timing",              xno_argument,       nullptr, 'T'},
            {"softmax",             xno_argument,       nullptr, 'S'},
            {"max-filesize",        xrequired_argument, nullptr, 'F'},
            {"no-extension-check",  xno_argument,       nullptr, 'D'},
            {"help",                xno_argument,       nullptr, 'h'},
            {"version",             xno_argument,       nullptr, 'v'},
            {"about",               xno_argument,       nullptr, 'a'},
            {0, 0, 0, 0} // Sentinel
        }};
    // clang-format on

    while(true)
    {
        auto const opt = xgetopt_long(argc, argv, short_opts.c_str(), long_options.data(), nullptr);

        if(opt == -1)
            break;

        // clang-format off
        switch(opt)
        {
            case 'm': result.model_path = xoptarg; break;
            case 'c': result.classes_path = xoptarg; break;
            case 'k': result.top_k = std::stoi(xoptarg); break;
            case 't': result.threads = std::stoi(xoptarg); break;
            case 'T': result.enable_timing = true; break;
            case 'S': result.use_softmax = true; break;
            case 'F': result.max_filesize = string_unit_to_numeric(xoptarg); break;
            case 'D': result.disable_extension_check = true; break;
            case 'h': print_help(); exit(EXIT_SUCCESS); break;
            case 'v': std::cout << PROJECT_VERSION << std::endl; exit(EXIT_SUCCESS); break;
            case 'a': print_about(); exit(EXIT_SUCCESS); break;
            default: throw std::runtime_error("could not parse parameters, use --help for usage.");
        }
        // clang-format on
    }

    // Process remaining non-option arguments
    for(int index = xoptind; index < argc; index++)
        result.image_files.push_back(argv[index]);

    if(result.threads == 0)
        result.threads = 1;

    return result;
}

/**
 * @brief The main worker thread function.
 *        Pops a file path from the input queue, performs classification,
 *        formats the result, and pushes it to the output queue.
 * @param tsq_in The thread-safe input queue for file paths.
 * @param tsq_out The thread-safe output queue for formatted results.
 * @param model The YOLO model instance to use for classification.
 * @param[in] c The application configuration.
 */
void thread_classify(tsqueue &tsq_in, tsqueue &tsq_out, yolo &model, configuration const &c)
{
    while(auto value = tsq_in.pop())
    {
        try
        {
            // Measure execution time
            auto start_timer = std::chrono::high_resolution_clock::now();

            // File path of the image
            auto const &path = *value;

            // Check if the path points to a regular file (not a directory, not non-existent)
            if(!std::filesystem::is_regular_file(path))
                throw std::filesystem::filesystem_error("Path is not a regular file or does not exist", path, std::make_error_code(std::errc::no_such_file_or_directory));

            // Check file size
            std::uintmax_t file_sz = std::filesystem::file_size(path);
            if(file_sz == 0)
                throw std::length_error("File is empty.");
            else if(file_sz > c.max_filesize)
                throw std::length_error("File is too large.");

            // Load the image
            cv::Mat image = cv::imread(path);

            if(image.empty())
                throw std::runtime_error("OpenCV could not read or decode image.");

            // Run the model and classify the image
            auto cls = model.predict(image, c.top_k);

            // Format result
            std::string result = path;

            if(c.enable_timing)
            {
                // Time of the image being loaded, resized and classified
                auto end      = std::chrono::high_resolution_clock::now();
                auto duration = end - start_timer;

                result += ", " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()) + "ms";
            }

            if(c.top_k != 0)
                result += ", ";

            for(auto it = cls.begin(); it != cls.end(); ++it)
            {
                result += it->class_name + " " + std::to_string(it->confidence);

                if(std::next(it) != cls.end())
                    result += ", ";
            }

            tsq_out.push(result);
        }
        catch(const std::exception &e)
        {
            std::stringstream ss;
            ss << "yolo-cls: could not process the file \'" << *value << "\': " << e.what() << std::endl;
            std::cerr << ss.str();
        }
    }
}

/**
 * @brief The output thread function.
 *        Pops formatted results from the output queue and prints them to standard output.
 * @param tsq The thread-safe output queue.
 */
void thread_print_tsq(tsqueue &tsq)
{
    while(auto value = tsq.pop())
    {
        std::cout << *value << std::endl;
    }
}

/**
 * @brief The input thread function for piped data.
 *        Reads lines (file paths) from standard input and pushes them to the input queue.
 * @param tsq_in The thread-safe input queue to push file paths to.
 * @param[in] c The application configuration (used for extension checking).
 */
void thread_get_line(tsqueue &tsq_in, configuration const &c)
{
    std::string line;
    while(std::getline(std::cin, line))
    {
        // Checking the file extention
        std::filesystem::path fp = line;
        std::string extension    = fp.extension().string();

        if(!c.disable_extension_check)
        {
            if(is_supported_image(extension))
                tsq_in.push(line);
        }
        else
            tsq_in.push(line);
    }
    tsq_in.close();
}

/**
 * @brief Prints help information that is invoked by `-h` or `--help`
*/
void print_help()
{
    std::string help =
        R"(yolo-cls: A command-line tool for YOLO-based image classification.

usage: yolo-cls [options...] [image_file...]
       <command> | yolo-cls [options...]

The application can process image file paths provided as arguments or piped from
standard input (one path per line).

Options:
  -m, --model <path>             Required. Path to the ONNX model file.
  -c, --classes <path>           Required. Path to the text file containing class names.
  -k, --top-k <int>              Number of top results to show. [default: 5]
  -t, --threads <int>            Number of threads to use for classification. [default: number of hardware cores]
  -F, --max-filesize <size>      Maximum allowed filesize for images (e.g., 100mb, 2g). [default: 100mb]
  -T, --timing                   Enable printing processing time for each image.
  -S, --softmax                  Apply softmax to the output scores.
  -D, --no-extension-check       Disable image file extension check (e.g., .jpg, .png).
  -h, --help                     Print this help message and exit.
  -v, --version                  Print version information and exit.
  -a, --about                    Print about information and exit.

Examples:
  yolo-cls -m ./yolo11x-cls.onnx -c ./imagenet.names ./fox.png
  find . | yolo-cls -m ./yolo11x-cls.onnx -c ./imagenet.names
)";

    std::cout << help << std::endl;
}

/**
 * @brief Prints about information that is invoked by `-a` or `--about`
*/
void print_about()
{
    std::string about = "yolo-cls: A command-line tool for YOLO-based image classification.\n"
                        "Version: " PROJECT_VERSION "\n"
                        "Author: " PROJECT_AUTHOR "\n"
                        "Homepage: " PROJECT_HOMEPAGE_URL "\n"
                        "License: " PROJECT_LICENSE;

    std::cout << about << std::endl;
}
