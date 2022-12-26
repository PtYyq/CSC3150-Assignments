#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void init_swap_table(VirtualMemory *vm){
  int table_size = vm->STORAGE_SIZE/vm->PAGESIZE;
  for (int i=0;i < table_size; i++){
    vm->swap_table[i] = 0; // 0 for empty
  }
}
// swap from disk to physical
__device__ void swap_in(VirtualMemory *vm, u32 page_num, int ind){
  int swap_table_size = vm->STORAGE_SIZE/vm->PAGESIZE;
  int disk_ind;
  // try to find page number in secondary memory
  for (int i=0;i<swap_table_size;i++){
    if (vm->swap_table[i]!=0){
      if (vm->swap_table[i+swap_table_size] == page_num){
        disk_ind = i;
        break;
      }
    }
  }
  // This case handles that page is not in both table
  if (disk_ind==swap_table_size){
    vm->invert_page_table[ind+vm->PAGE_ENTRIES] = page_num;
    vm->invert_page_table[ind] &= 0x7fffffff; // update valid bit
    return;
  }

  for (int i=0;i<vm->PAGESIZE;i++){
    vm->buffer[ind*vm->PAGESIZE+i] = vm->storage[disk_ind*vm->PAGESIZE+i];
  }
  vm->swap_table[disk_ind] = 0; // update swap table
  // update inverted page table
  vm->invert_page_table[ind+vm->PAGE_ENTRIES] = page_num;
  vm->invert_page_table[ind] &= 0x7fffffff; // update valid bit
}
// swap from physical to disk
__device__ void swap_out(VirtualMemory *vm, u32 page_num, int ind){
  int swap_table_size = vm->STORAGE_SIZE/vm->PAGESIZE;
  int disk_ind;
  // try to find a empty place in secondary memory
  for (int i=0;i<swap_table_size;i++){
    if (vm->swap_table[i]==0){
      disk_ind = i;
      break;
    }
  }
  for (int i=0;i<vm->PAGESIZE;i++){
    vm->storage[disk_ind*vm->PAGESIZE+i] = vm->buffer[ind*vm->PAGESIZE+i];
  }
  // update swap table
  vm->swap_table[disk_ind] = 1;
  vm->swap_table[disk_ind+swap_table_size] = page_num;
}

// update the counter
__device__ void update_counter(VirtualMemory *vm, int frame_ind){
      for (int i=0;i<vm->PAGE_ENTRIES;i++){
        vm->invert_page_table[i+2*vm->PAGE_ENTRIES]+=1;
      }
      vm->invert_page_table[frame_ind+2*vm->PAGE_ENTRIES] = 0;
}

// find victim frame index
__device__ int victim_index(VirtualMemory *vm){
  int max = vm->invert_page_table[2*vm->PAGE_ENTRIES];
  int index = 0;
  for (int i=1;i<vm->PAGE_ENTRIES;i++){
    if (vm->invert_page_table[2*vm->PAGE_ENTRIES+i]>max){
      max = vm->invert_page_table[2*vm->PAGE_ENTRIES+i];
      index = i;
    }
  }
  return index;
}

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
    vm->invert_page_table[i + 2*vm->PAGE_ENTRIES] = 0; // a counter for LRU algorithm
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, u32 *swaptable, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;
  vm->swap_table = swaptable;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
  init_swap_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  u32 page_num = addr / vm->PAGESIZE;
  u32 offset = addr - page_num * vm->PAGESIZE;
  int pEntries = vm->PAGE_ENTRIES;
  int pSize = vm->PAGESIZE;
  int frame_ind = -1;
  int empty_frame_ind = -1;
  // try to find empty frame
  for (int i=0;i<pEntries;i++){
    if (vm->invert_page_table[i]>>31){
      empty_frame_ind = i;
      break;
    }
  }
  // try to find the corresponding frame
  for (int i=0;i<pEntries;i++){
    if (!(vm->invert_page_table[i]>>31)){
      if (vm->invert_page_table[i+pEntries] == page_num){
        frame_ind = i;
        break;
      }
    }
  }

  // find a frame in inverted page table
  if (frame_ind != -1){
    update_counter(vm,frame_ind);
    return vm->buffer[frame_ind*pSize+offset];
  }
  // cannot find such frame
  else{
    *(vm->pagefault_num_ptr)+=1;
    if (empty_frame_ind != -1){
      swap_in(vm, page_num, empty_frame_ind); // swap in the page from secondary memory
      update_counter(vm,empty_frame_ind);
      return vm->buffer[empty_frame_ind*pSize+offset];
    }
    else{
      int victim_ind = victim_index(vm);
      u32 victim_pn = vm->invert_page_table[victim_ind+pEntries];
      swap_out(vm,victim_pn,victim_ind);  // swap out the victim frame to secondary memory
      swap_in(vm,page_num,victim_ind);  // swap in the page from secondary memory
      update_counter(vm,victim_ind);
      return vm->buffer[victim_ind*pSize+offset];
    }
  }
  return 123; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  u32 page_num = addr / vm->PAGESIZE;
  u32 offset = addr - page_num * vm->PAGESIZE;
  int pEntries = vm->PAGE_ENTRIES;
  int pSize = vm->PAGESIZE;
  int frame_ind = -1;
  int empty_frame_ind = -1;
  // try to find empty frame
  for (int i=0;i<pEntries;i++){
    if (vm->invert_page_table[i]>>31){
      empty_frame_ind = i;
      break;
    }
  }
  // try to find corresponding frame
  for (int i=0;i<pEntries;i++){
    if (!(vm->invert_page_table[i]>>31)){
      if (vm->invert_page_table[i+pEntries] == page_num){
        frame_ind = i;
        break;
      }
    }
  }

  // find such frame in inverted page table
  if (frame_ind != -1){
    update_counter(vm,frame_ind);
    vm->buffer[frame_ind*pSize+offset] = value;
  }
  else{
    *(vm->pagefault_num_ptr)+=1;
    if (empty_frame_ind != -1){
      swap_in(vm, page_num, empty_frame_ind); // swap in the page from secondary memory
      update_counter(vm,empty_frame_ind);
      vm->buffer[empty_frame_ind*pSize+offset] = value;
    }
    else{
      int victim_ind = victim_index(vm);
      u32 victim_pn = vm->invert_page_table[victim_ind+pEntries];
      swap_out(vm,victim_pn,victim_ind);  //swap out the victim frame to secondary memory
      swap_in(vm,page_num,victim_ind);  // swap in the page from secondary memory
      update_counter(vm,victim_ind);
      vm->buffer[victim_ind*pSize+offset] = value;
    }
  }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
   for (int i=offset;i<input_size+offset;i++){
    results[i-offset] = vm_read(vm,i);
   }
}