#pragma once
#include <string>
#include <vector>

class MyProfile
{
public:
    MyProfile() = delete;
    MyProfile(const std::string &name, const std::vector<std::pair<std::string, std::string>> &args = std::vector<std::pair<std::string, std::string>>());
    ~MyProfile();

private:
    std::string _name;
    uint64_t _ts1;
    std::vector<std::pair<std::string, std::string>> _args;
};

#define MY_PROFILE(NAME) MyProfile(NAME + std::string(":") + std::to_string(__LINE__))
#define MY_PROFILE_ARGS(NAME, ...) MyProfile(NAME + std::string(":") + std::to_string(__LINE__), __VA_ARGS__)

#define SIMPLE_PROFILE(NAME) \
        auto p = MyProfile(NAME + std::string(":") + std::to_string(__LINE__));

// #define SIMPLE_PROFILE(NAME)

// Example 1: MY_PROFILE / MY_PROFILE_ARGS
/******************************************************
auto p = MY_PROFILE("fun_name")
Or
{
    auto p = MY_PROFILE("fun_name")
    func()
}
Or
{
    auto p2 = MY_PROFILE_ARGS("fun_name", {{"arg1", "sleep 30 ms"}});
    func()
}
******************************************************/

#define MY_PROFILE_VAR(VAR, NAME) auto VAR = MY_PROFILE(NAME)
#define MY_PROFILE_VAR_ARGS(VAR, NAME, ...) auto VAR = MY_PROFILE_ARGS(NAME, __VA_ARGS__)
// Example 2: MY_PROFILE / MY_PROFILE_ARGS
/******************************************************
MY_PROFILE_VAR(p, "fun_name")
Or
{
    MY_PROFILE_VAR(p1, "fun_name")
    func()
}
Or
{
    MY_PROFILE_VAR_ARGS(p2, "fun_name", {{"arg1", "sleep 30 ms"}});
    func()
}
******************************************************/