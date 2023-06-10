#include "timer.h"
#ifdef __MPI
#include <mpi.h>
#endif
#include <cassert>

#ifndef __MPI
timer::func_timer::func_timer(func name)
{
    cur_start = steady_clock::now();
    func_name = name;
    calls = 0;
    triggered = false;
    duration = cur_start - cur_start;
}
timer::func_timer::func_timer()
{
    cur_start = steady_clock::now();
    func_name = {"", ""};
    calls = 0;
    triggered = false;
    duration = cur_start - cur_start;
}
void timer::func_timer::tick()
{
    if (triggered)
    {
        duration += (steady_clock::now() - cur_start);
        triggered = false;
    }
    else
    {
        cur_start = steady_clock::now();
        ;
        triggered = true;
        calls++;
    }
}
double timer::func_timer::get_duration()
{
    return (double)duration_cast<microseconds>(duration).count() / 1e6;
}
#else
timer::func_timer::func_timer(func name)
{
    cur_start = MPI_Wtime();
    func_name = name;
    calls = 0;
    triggered = false;
    duration = cur_start - cur_start;
}
timer::func_timer::func_timer()
{
    cur_start = MPI_Wtime();
    func_name = {"", ""};
    calls = 0;
    triggered = false;
    duration = cur_start - cur_start;
}
timer::func_timer::func_timer(timer::timer_pack pack)
{
    func_name = std::make_pair(pack.class_name, pack.func_name);
    duration = pack.dura;
    calls = pack.calls;
    triggered = false;
}
void timer::func_timer::tick()
{
    if (triggered)
    {
        duration += (MPI_Wtime() - cur_start);
        triggered = false;
    }
    else
    {
        cur_start = MPI_Wtime();
        ;
        triggered = true;
        calls++;
    }
}
double timer::func_timer::get_duration()
{
    return duration;
}
#endif
int timer::func_timer::get_calls()
{
    return calls;
}
func timer::func_timer::get_name()
{
    return func_name;
}

std::map<func, timer::func_timer> timer::name_map = std::map<func, func_timer>();

void timer::tick(const std::string &class_name, const std::string &func_name)
{
    func tmp = {class_name, func_name};
    auto class_iter = name_map.find(tmp);
    if (class_iter != name_map.end())
    {
        (*class_iter).second.tick();
    }
    else
    {
        name_map[tmp] = func_timer(tmp);
        name_map[tmp].tick();
    }
}
void timer::print()
{
    #ifdef __MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
    {
        return;
    }
    #endif
    std::cout << "|=Class_Name==========|=Func_Name===========|Calls=|=Time(sec)==|=Avg(sec)===|=Per%====|"
              << std::endl;
    double total = name_map[{"", "total"}].get_duration();
    for (auto iter = name_map.begin(); iter != name_map.end(); iter++)
    {
        func name = (*iter).first;
        func_timer &tmr = (*iter).second;
        std::cout.setf(std::ios::right);
        std::cout << "|" << std::setw(20) << name.first << " |" << std::setw(20) << name.second << " |" << std::setw(5)
                  << tmr.get_calls() << " |" << std::setw(11) << std::setprecision(6) << std::defaultfloat
                  << tmr.get_duration() << " |" << std::setw(11) << std::defaultfloat << std::setprecision(6)
                  << tmr.get_duration() / tmr.get_calls() << " |" << std::setw(7) << std::fixed << std::setprecision(1)
                  << tmr.get_duration() / total * 100 << "% |" << std::endl;
    }
    std::cout << "|=====================|=====================|======|============|============|=========|"
              << std::endl;
}
#ifdef __MPI
timer::timer_pack::timer_pack(std::pair<func, func_timer> pair)
{
    dura = pair.second.get_duration();
    calls = pair.second.get_calls();
    std::string _cla = pair.first.first, _fun = pair.first.second;
    assert(_cla.size() < 32 && _fun.size() < 32);
    
    strcpy(class_name, _cla.c_str());
    strcpy(func_name, _fun.c_str());
}
timer::timer_pack::timer_pack()
{
    calls = 0;
    dura = 0;
}
void timer::mpi_sync()
{
    MPI_Barrier(MPI_COMM_WORLD);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int block_lengths[4] = {32, 32, 1, 1};
    MPI_Aint offsets[4] = {offsetof(timer::timer_pack, class_name), offsetof(timer::timer_pack, func_name), offsetof(timer::timer_pack, dura), offsetof(timer::timer_pack, calls)};
    MPI_Datatype types[4] = {MPI_CHAR, MPI_CHAR, MPI_DOUBLE, MPI_INT};
    MPI_Datatype mpi_timer_type;
    MPI_Type_create_struct(4, block_lengths, offsets, types, &mpi_timer_type);
    MPI_Type_commit(&mpi_timer_type);
    for (int i = 1; i < size; i ++)
    {    
        if (rank == i)
        {
            int send_size = name_map.size(), recv_size;
            MPI_Send(&send_size, 1, MPI_INT, 0, (i+1)*(0+1), MPI_COMM_WORLD);
            timer_pack *packs = new timer_pack[send_size];
            int cnt = 0;
            for (auto it = name_map.begin(); it != name_map.end(); it++)
            {
                packs[cnt] = timer_pack(*it);
                cnt ++;
            }
            // std::cout << "packs sent from " << rank << " with size " << send_size << std::endl;
            bool succ = MPI_Send(packs, send_size, mpi_timer_type, 0, 2*(i+1)*(0+1), MPI_COMM_WORLD);
        }
        if (rank == 0)
        {
            int recv_size;
            MPI_Recv(&recv_size, 1, MPI_INT, i, (i+1)*(0+1), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            timer_pack *packs;

            if (recv_size > 0)
            {
                // std::cout << ' ' << recv_size << std::endl;
                packs = new timer_pack[recv_size];
                MPI_Recv(packs, recv_size, mpi_timer_type, i, 2*(i+1)*(0+1), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // std::cout << status.MPI_ERROR << std::endl;
                
                // std::cout << packs[0].class_name << packs[0].func_name << std::endl;
            }
            for (int i = 0; i < recv_size; i++)
            {

                func_timer tmp(packs[i]);
                name_map.insert(std::make_pair(tmp.get_name(), tmp));
            }
            // std::cout << "recv exited" << std::endl;
        }
    }
    
    // MPI_Wait(&request_recv_size, MPI_STATUS_IGNORE);
    // MPI_Wait(&request_recv_pack, MPI_STATUS_IGNORE);
}
#endif