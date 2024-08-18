# GPU-Accelerated-Scene-Translation
Developed a **parallel algorithm** using **CUDA** to achieve real-time scene translation of grayscale images, achieving a **40x performance** benefit compared to sequential CPU-based execution while handling up to **10 million images and translations**.

Leveraged **CSR (Compressed Sparse Row)** representation to efficiently manage and traverse the scene graph using **parallel BFS**, efficiently determining transitive dependencies for accurate hierarchical translation of dependent images.

Exploited **shared memory** and **memory coalescing** for optimized data access, managed thread **synchronization** with **atomic based locking** protocols, and extensively utilized CUDA's **Thrust libraries** for parallel operations, including reduction, sorting to speed up computation.
