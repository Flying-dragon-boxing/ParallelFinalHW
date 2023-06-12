#ifndef FUNC_TIMER_HPP
#define FUNC_TIMER_HPP
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#ifndef __MPI
#include <chrono>
using namespace std::chrono;
#else
#include <mpi.h>
#endif

typedef std::pair<std::string, std::string> func;

class timer
{
  class func_timer;
#ifdef __MPI
  struct timer_pack
  {
    char class_name[32], func_name[32];
    double dura;
    int calls;
    timer_pack(std::pair<func, func_timer>);
    timer_pack();
  };
#endif
    class func_timer
    {
      private:
#ifndef __MPI
        steady_clock::time_point cur_start; // start time of current count
        steady_clock::duration duration; // duration of the function call
#else
        double cur_start;
        double duration;
#endif
        func func_name; // class name and func name
        unsigned long long calls; // time of calls, unsigned long long
        bool triggered; // whether the timer iis triggered

      public:
        func_timer(func name); // constructor
        func_timer();
#ifdef __MPI
        func_timer(timer::timer_pack);
#endif
        void tick(); // tick
        double get_duration(); // duration time in second
        int get_calls(); // time of calls
        func get_name();
    };

  private:
    static std::map<func, func_timer> name_map; // name to func_timer map

  public:
    // static void init();
    static void tick(const std::string &class_name,
                     const std::string &func_name); // tick the timer of given class and func name
    static void print(); // print timer stats
#ifdef __MPI
    static void mpi_sync(MPI_Comm comm = MPI_COMM_WORLD); // sync timer stats
#endif
};

#endif