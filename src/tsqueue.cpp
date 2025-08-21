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
 * @file tsqueue.cpp
 * @brief Defines a simple thread-safe queue.
 * @author Savelii Pototskii
 * @date 2025-08-16
 * @copyright Copyright (C) 2025 Savelii Pototskii (savalione.com)
 * @copyright SPDX-License-Identifier: GPL-3.0-or-later
*/
#include "tsqueue.h"

/**
 * @brief Pushes a string onto the queue in a thread-safe manner.
 * @param[in] value The string to push.
 */
void tsqueue::push(std::string const &value)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(value);
    }
    cv.notify_one();
}

/**
 * @brief Pops a string from the queue. This operation is blocking.
          It will wait until an item is available or until the queue is closed.
 * @return An `std::optional<std::string>` containing the value if one was popped, or `std::nullopt` if the queue is empty and has been closed.
 */
std::optional<std::string> tsqueue::pop()
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return !queue.empty() || done; });

    if(queue.empty())
    {
        return std::nullopt;
    }

    std::string value = std::move(queue.front());
    queue.pop();
    return value;
}

/**
 * @brief Closes the queue, signaling that no more items will be pushed. This will unblock any threads waiting on `pop()`.
 */
void tsqueue::close()
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
    }

    cv.notify_all();
}