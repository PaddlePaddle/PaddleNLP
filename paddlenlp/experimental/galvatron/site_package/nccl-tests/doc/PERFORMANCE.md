# Performance reported by NCCL tests

NCCL tests report the average operation time in ms, and two bandwidths in GB/s : algorithm bandwidth and bus bandwidth. This page explains what those numbers mean and what you should expect depending on the hardware used.

# Time

Time is useful with small sizes, to measure the constant overhead (or latency) associated with operations.

On large sizes, the time becomes linear with the size (since it is roughly equal to overhead + size / bw) and is no longer measuring the latency but
also the bandwidth multiplied by the size.

Therefore, on large sizes, it makes more sense to look at the bandwidth.

# Bandwidth

## Algorithm bandwidth

Algorithm bandwidth is using the most commonly used formula for bandwidth : size (_S_) / time (_t_). It is useful to compute how much time any large operation would take by simply dividing the size of the operation by the algorithm bandwidth.

`algbw = S/t`

## Bus bandwidth

While the algorithm bandwidth makes sense for point-to-point operations like Send/Receive, it is not always helpful to measure collective operations speed, since the theoretical peak algorithm bandwidth is not equal to the hardware peak bandwidth, usually depending on the number of ranks.
Most benchmarks only provide time measurements, which is hard to interpret for large sizes. Some others also provide algorithms bandwidth, but see that depending on the number of ranks, that bandwidth varies (and decreases as the number of ranks increase).

To provide a number which reflects how optimally the hardware is used, NCCL tests introduce the notion of "Bus Bandwidth" ("busbw" column in the tests output).
This number is obtained applying a formula to the algorithm bandwidth to reflect the speed of the inter-GPU communication.
Using this bus bandwidth, we can compare it with the hardware peak bandwidth, independently of the number of ranks used.

The formula depends on the collective operation.

### AllReduce

An allreduce operation, for each element of the N arrays (input i_X and output o_X, each situated on rank X), is performing the following operation :

`o_0 = o_1 = o_2 = ... = o_{n-1} = i_0 + i_1 + i_2 + ... + i_{n-1}`

**Note : this is independent of the algorithm used (ring, tree, or other) as long as they use point-to-point operations (send/receive).**

A ring would do that operation in an order which follows the ring :

`i_0 + i_1 + ... + i_{n-1} -> o_{n-1} -> o_0 -> o_1 -> .. -> o_{n-2}`

A tree would do it hierarchically :

`(((((i_{n-1} + i_{n-2}) + (i_{n-3} + i_{n-4})) + ... + (i_1 + i_0))))) -> o_0 -> (o_{n/2} -> (o_{3n/4} ...))`

In all cases, we need n-1 additions and n assignments for each element. Since every step is on a different rank except potentially one (the last input and the first output),
we need 2(n-1) data transfers (x number of elements) to perform an allReduce operation.

Considering that each rank has a bandwidth to the outside world of _B_, the time to perform an allReduce operation of _S_ elements is at best :

 `t = (S*2*(n-1)) / (n*B)`

Indeed, we have _S_ elements, 2*(n-1) operations per element, and _n_ links of bandwidth _B_ to perform them.
Reordering the equation, we find that

 `t = (S/B) * (2*(n-1)/n)`

Therefore, to get an AllReduce bandwidth measurement which we can compare to the hardware peak bandwidth, we compute :

 `B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)`

### ReduceScatter

The ReduceScatter operation requires only to perform the addition part of the allReduce operation :

 `o_K = i_0 + i_1 + i_2 + ... + i_{n-1}`

With K being the rank which is getting the final result(K=offset/recvsize).

The perfect reduceScatter time with a rank bandwidth of B would therefore be :

 `t = S*(n-1) / (B*n)`

And the Bus Bandwidth is therefore computed as :

 `B = S/t * (n-1)/n = algbw * (n-1)/n`

Note that here, S is the size in bytes of the total array, which for NCCL is equal to `recvcount*sizeof(datatype)*n` as the `recvcount` argument is the count per rank.

### AllGather

The AllGather operation requires only to perform the assignment part of the allReduce operation :

 `o_0 = o_1 = o_2 = ... = o_{n-1} = i_K`

With K being the rank where the data originates from (K=offset*sendsize).

The perfect allGather time with a rank bandwidth of B would therefore be :

 `t = S*(n-1) / (B*n)`

And the Bus Bandwidth is therefore computed as :

 `B = S/t * (n-1)/n = algbw * (n-1)/n`

Note that here, S is the size in bytes of the total array, which for NCCL is equal to `sendcount*sizeof(datatype)*n` as the `sendcount` argument is the count per rank.

### Broadcast

The broadcast operation representation is similar to allGather :

 `o_0 = o_1 = o_2 = ... = o_{n-1} = i_R`

R being the root of the operation.

However, in this case, since the i_R input is not evenly distributed on the ranks, we cannot use all N links to perform the transfer operations.
Indeed, *all* data has to get out of the root rank, hence the bottleneck is on the root rank which only has B as capacity to get data out :

 `t = S/B`

And :

 `B = S/t`

### Reduce

The reduce operation performs :

 `o_R = i_0 + i_1 + i_2 + ... + i_{n-1}`

R being the root of the operation.

Similarly to broadcast, all data need to be sent to the root, hence :

 `t = S/B`

And :

 `B = S/t`

### Summary

To obtain a bus bandwidth which should be independent of the number of ranks _n_, we apply a correction factor to the algorithm bandwidth :

* AllReduce : 2*(_n_-1)/_n_
* ReduceScatter : (_n_-1)/_n_
* AllGather : (_n_-1)/_n_
* Broadcast : 1
* Reduce : 1

The bus bandwidth should reflect the speed of the hardware bottleneck : NVLink, PCI, QPI, or network.
