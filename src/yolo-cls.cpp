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
 * @file yolo-cls.cpp
 * @brief The yolo-cls implementation.
 * @details
 *
 * yolo-cls is a command-line tool for YOLO-based image classification.
 *
 * The application can process image file paths provided as arguments or piped from
 * standard input (one path per line).
 *
 * @author Savelii Pototskii
 * @date 2025-08-17
 * @copyright Copyright (C) 2025 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
*/
#include <unistd.h> // For unix pipe

#include "utils.h"

int main(int argc, char **argv)
{
    // Application configuration
    configuration config;

    // Check options
    try
    {
        config = parse_arguments(argc, argv);
    }
    catch(std::exception const &e)
    {
        std::stringstream ss;
        ss << "yolo-cls: " << e.what() << std::endl;
        std::cerr << ss.str();

        return EXIT_FAILURE;
    }

    // Create classifier
    yolo classifier;

    // Initialize classifier
    try
    {
        classifier = yolo(config.model_path, config.classes_path, config.use_softmax);
    }
    catch(std::exception const &e)
    {
        std::stringstream ss;
        ss << "yolo-cls: " << e.what() << std::endl;
        std::cerr << ss.str();

        return EXIT_FAILURE;
    }

    // Thread safe queues for input/output
    tsqueue tsq_in;
    tsqueue tsq_out;

    // Run piped output in a single separate thread
    std::thread output_thread(thread_print_tsq, std::ref(tsq_out));

    // Create worker threads for classification
    std::vector<std::thread> worker_threads;
    for(int i = 0; i < config.threads; ++i)
    {
        worker_threads.emplace_back(thread_classify, std::ref(tsq_in), std::ref(tsq_out), std::ref(classifier), std::ref(config));
    }

    // Check whether the executable is invoked by a unix pipe or not
    if(isatty(STDIN_FILENO))
    {
        if(config.image_files.empty())
        {
            // No image files provided and no piped input
        }

        // Add images to the thread safe input queue
        for(auto const &i : config.image_files)
            tsq_in.push(i);

        // Close the queue because there won't be any input
        tsq_in.close();
    }
    else
    {
        // Input from a pipe
        std::thread input_thread(thread_get_line, std::ref(tsq_in), std::ref(config));

        // Wait until the end of the piped input
        input_thread.join();
    }

    // Wait for worker threads to finish processing all items
    for(std::thread &t : worker_threads)
    {
        t.join();
    }

    // Signal that no more output will be generated
    tsq_out.close();

    // Wait for the output thread to finish printing
    output_thread.join();

    return EXIT_SUCCESS;
}
