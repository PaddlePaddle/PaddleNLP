#pragma once
#include <iostream>

/// @brief Defines a macro to disable log output.
/// @details This macro returns a `NullStream` object, and all content passed via `<<` is ignored.
///          It is designed to keep the code compilable without modifying log call statement.
/// @param level The log level (ignored).
class NullStream {
public:
    template <typename T>
    NullStream& operator<<(const T&) { return *this; }
};

#define VLOG(level) NullStream()