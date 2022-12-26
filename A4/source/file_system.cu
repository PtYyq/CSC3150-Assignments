#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
}
// helper function to get and store information of FCB
// The information contains:
// valid bit: 31 --> 0 for invalid, 1 for valid
// file address: 28-30 address from 0 - 2**23 so 24 bits is enough
// file size: 26-27 --> max file size is 2**10 bits, needs 10 bits, so 16 bits can handle
// time: 22-25 --> create: 22-23; modified: 24-25

__device__ u32 getAdd(FileSystem *fs,int ind);
__device__ void storeAdd(FileSystem *fs,int ind,u32 add);
__device__ u32 getSize(FileSystem *fs,int ind);
__device__ void storeSize(FileSystem *fs,int ind,u32 size);
__device__ u32 getCreateTime(FileSystem *fs,int ind);
__device__ void storeCreateTime(FileSystem *fs,int ind,u32 time);
__device__ u32 getModifiedTime(FileSystem *fs,int ind);
__device__ void storeModifiedTime(FileSystem *fs,int ind,u32 time);
__device__ int getNameInd(FileSystem *fs,char *s);
__device__ void storeName(FileSystem *fs,int ind,char *s);

__device__ int getEmptyBlock(FileSystem *fs); // function to get empty block
__device__ int getEmptyFCB(FileSystem *fs); // function to get empty FCB
__device__ void setSuper(FileSystem *fs,int blockInd,int target); // set number of super block
__device__ int getSuper(FileSystem *fs,int blockInd); // get number of super block
__device__ void moveBlock(FileSystem *fs, int originInd, int targetInd); // move the block from one place to another
__device__ void compaction(FileSystem *fs); // function to compact the file system


__device__ u32 getAdd(FileSystem *fs,int ind){
  u32 result = 0;
  for (int i=0;i<3;i++){
    result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+28+i]<<(8*i));
  }
  return result;
}
__device__ void storeAdd(FileSystem *fs,int ind,u32 add){
  for (int i=0;i<3;i++){
    // u32 temp = add>>(8*i);
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+28+i] = add>>(8*i);
  }
}
__device__ u32 getSize(FileSystem *fs,int ind){
  u32 result = 0;
  for (int i=0;i<2;i++){
    result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+26+i]<<(8*i));
  }
  return result;
}
__device__ void storeSize(FileSystem *fs,int ind,u32 size){
  for (int i=0;i<2;i++){
    // u32 temp = (size>>(8*i)) & 0xff;
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+26+i] = size>>(8*i);
  }
}
__device__ u32 getCreateTime(FileSystem *fs,int ind){
  u32 result = 0;
  for (int i=0;i<2;i++){
    result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+22+i]<<(8*i));
  }
  return result;
}
__device__ u32 getModifiedTime(FileSystem *fs,int ind){
  u32 result = 0;
  for (int i=0;i<2;i++){
    result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+24+i]<<(8*i));
  }
  return result;
}
__device__ void storeCreateTime(FileSystem *fs,int ind,u32 time){
  for (int i=0;i<2;i++){
    u32 temp = (time>>(8*i)) & 0xff;
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+22+i] = temp;
  }
}
__device__ void storeModifiedTime(FileSystem *fs,int ind,u32 time){
  for (int i=0;i<2;i++){
    u32 temp = (time>>(8*i)) & 0xff;
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+24+i] = temp;
  }
}

__device__ int getNameInd(FileSystem *fs,char *s){
    int ind = -1;
    int superSize = fs->SUPERBLOCK_SIZE;
    for (int i=0;i<fs->FCB_ENTRIES;i++){
      if (fs->volume[superSize+i*fs->FCB_SIZE+31]==0){
        continue;
      }
      for (int j=0;j<fs->MAX_FILENAME_SIZE;j++){
        if (fs->volume[superSize+i*fs->FCB_SIZE+j] != s[j]){
          break;
        }
        if (s[j]=='\0'){
          return i;
        }
      }
    }
    return -1;
}
__device__ char* getName(FileSystem *fs,int ind){
  char result[20];
  for (int i=0;i<fs->MAX_FILENAME_SIZE;i++){
    result[i] = fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+i];
    if (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+i]=='\0'){
      break;
    }
  }
  return result;
}
__device__ void storeName(FileSystem *fs,int ind,char *s){
  for (int i=0;i<fs->MAX_FILENAME_SIZE;i++){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+i] = s[i];
    if (s[i]=='\0'){
      break;
    }
  }
}
__device__ int getEmptyBlock(FileSystem *fs){
  for (int i=0;i<fs->SUPERBLOCK_SIZE*8;i++){
      if (getSuper(fs,i)==0){
        return i;
      }
    }
  return -1;
}
__device__ int getEmptyFCB(FileSystem *fs){
  for (int i = 0;i<fs->FCB_ENTRIES;i++){
    if (fs->volume[fs->SUPERBLOCK_SIZE+i*fs->FCB_SIZE+31] == 0){
      return i;
    }
  }
  return -1;
}
__device__ void setSuper(FileSystem *fs,int blockInd,int target){
  if (blockInd >= fs->SUPERBLOCK_SIZE*8){
    printf("Block out of index!\n");
    return;
  }
  int s = blockInd / 8;
  int remain = blockInd % 8;
  if (target == 0){
    fs->volume[s] &= (~(1<<remain));
  }
  if (target == 1){
    fs->volume[s] |= (1<<remain);
  }
}
__device__ int getSuper(FileSystem *fs,int blockInd){
  if (blockInd >= fs->SUPERBLOCK_SIZE*8){
    printf("Block out of index!\n");
    return -1;
  }
  int s = blockInd / 8;
  int remain = blockInd % 8;
  return (fs->volume[s]>>remain)&0x01;
}

__device__ void moveBlock(FileSystem *fs, int originInd, int targetInd){
  if (getSuper(fs,targetInd)==1){
    printf("Cannot move to a non empty block!\n");
    return;
  }
  u32 originAdd = originInd * 32;
  u32 targetAdd = targetInd * 32;
  for (int i=0;i<fs->STORAGE_BLOCK_SIZE;i++){
    fs->volume[fs->FILE_BASE_ADDRESS+targetAdd+i] = fs->volume[fs->FILE_BASE_ADDRESS+originAdd+i];
  }
  setSuper(fs,originInd,0);
  setSuper(fs,targetInd,1);
}
__device__ void compaction(FileSystem *fs){
  // move the block content
  // first find the empty block
  int emptyBlock = -1;
  for (int i=0;i<fs->SUPERBLOCK_SIZE*8;i++){
      if (getSuper(fs,i)==0){
        emptyBlock = i;
        break;
      }
  }
  if (emptyBlock==-1){
    printf("No empty block!\n");
    return;
  }
  // then find the next block that is not empty
  int nextUnempty = -1;
  for (int i=emptyBlock+1;i<fs->SUPERBLOCK_SIZE*8;i++){
    if (getSuper(fs,i)==1){
      nextUnempty = i;
      break;
    }
  }
  if (nextUnempty==-1){
    // the situation that no need to compact
    return;
  }
  // find the unempty terminates
  int unemptyEnd = -1;
  for (int i=nextUnempty+1;i<fs->SUPERBLOCK_SIZE*8;i++){
    if (getSuper(fs,i)==0){
      unemptyEnd = i;
      break;
    }
  }
  if (unemptyEnd == -1){
    unemptyEnd = fs->SUPERBLOCK_SIZE*8;
  }
  // move the unempty block to fill the empty block
  int holeSize = nextUnempty - emptyBlock;
  for (int i = nextUnempty;i<unemptyEnd;i++){
    moveBlock(fs,i,i-holeSize);
  }
  // Update FCB
  for (int i=0;i<fs->FCB_ENTRIES;i++){
    if (fs->volume[fs->SUPERBLOCK_SIZE+i*fs->FCB_SIZE+31]==0){
      continue;
    }
    u32 fileAdd = getAdd(fs,i);
    int fileStartBlock = fileAdd / 32;
    // only update the file that need to move
    if (fileStartBlock >= nextUnempty){
      int updatedBlockInd = fileStartBlock - holeSize;
      u32 updatedAdd = updatedBlockInd*32;
      storeAdd(fs,i,updatedAdd);
    }
  }
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  gtime += 1;
  int ind = getNameInd(fs,s);
  // case that the file does not exist
  if (ind == -1){
    if (op == G_READ){
      printf("Can not open the file %s\n",s);
      return -1;
    }
    else if (op == G_WRITE){
      int emptyFCB = getEmptyFCB(fs);
      if (emptyFCB==-1){
        printf("File number over 1024, can not open new file!\n");
        return -1;
      }
      else{
        u32 emptyBlock = getEmptyBlock(fs);
        if (emptyBlock==-1){
          printf("No enough space!");
          return -1;
        }
        fs->volume[fs->SUPERBLOCK_SIZE+emptyFCB*fs->FCB_SIZE+31] = 1;
        storeName(fs,emptyFCB,s);
        storeAdd(fs,emptyFCB,emptyBlock*fs->STORAGE_BLOCK_SIZE);
        storeSize(fs,emptyFCB,0);
        storeCreateTime(fs,emptyFCB,gtime);
        storeModifiedTime(fs,emptyFCB,gtime);
        return emptyFCB;
      }
    }
  }
  // case that the file does not exist
  else{
    return ind;
  }
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  gtime+=1;
  if (fp==-1){
    printf("Invalid file pointer!\n");
    return;
  }
  u32 add = getAdd(fs,fp);
  u32 fileSize = getSize(fs,fp);
  if (size>fileSize){
    printf("Read size overflow!\n");
    return;
  }
  for (u32 i=0;i<size;i++){
    output[i] = fs->volume[fs->FILE_BASE_ADDRESS+add+i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
  gtime+=1;
  if (fp==-1){
    printf("Invalid file pointer!\n");
    return -1;
  }
  u32 add = getAdd(fs,fp);
  u32 fileSize = getSize(fs,fp);
  if (fs->volume[fs->SUPERBLOCK_SIZE+fp*fs->FCB_SIZE+31]==1){
    // clean the origin content
    if (fileSize!=0){
      int blockNum = fileSize/32;
      if (fileSize%32!=0){
        blockNum += 1;
      }
      int startBlock = add / 32;
      for (int i=0;i<blockNum;i++){
        setSuper(fs,startBlock+i,0);
      }
    }

    // do a compaction
    compaction(fs);

    // now get the empty space and write the content
    int emptyBlock = getEmptyBlock(fs);
    if (emptyBlock==-1){
      printf("No enough space!");
      return -1;
    }
    else{
      add = emptyBlock*32;
      for (u32 i=0;i<size;i++){
        fs->volume[fs->FILE_BASE_ADDRESS+add+i] = input[i];
      }
      // update FCB
      storeAdd(fs,fp,add);
      storeSize(fs,fp,size);
      storeModifiedTime(fs,fp,gtime);
      // update super block
      int blockNum = size/32;
      if (size%32!=0){
        blockNum += 1;
      }
      for (int i=0;i<blockNum;i++){
        setSuper(fs,emptyBlock+i,1);
      }
    }
  }
  else{
    printf("Invalid file!\n");
    return -1;
  }
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  gtime+=1;
  int entries[1024];
  int len = 0;
  for (int i=0;i<fs->FCB_ENTRIES;i++){
    if (fs->volume[fs->SUPERBLOCK_SIZE+i*fs->FCB_SIZE+31]==1){
      entries[len] = i;
      len+=1;
    }
  }
  // insertion sort to sort the file
  if (op == LS_D){
    for (int i=1;i<len;i++){
      int j = i;
      while(getModifiedTime(fs,entries[j])>getModifiedTime(fs,entries[j-1])){
        int temp = entries[j];
        entries[j] = entries[j-1];
        entries[j-1] = temp;
        j-=1;
        if (j<1){
          break;
        }
      }
    }
    printf("===sort by modified time===\n");
		for (int i = 0; i < len; i++) {
			printf("%s\n", getName(fs,entries[i]));
		}
  }
  else if (op == LS_S){
    for (int i=1;i<len;i++){
      int j = i;
      while(getSize(fs,entries[j])>=getSize(fs,entries[j-1])){
        if (getSize(fs,entries[j])==getSize(fs,entries[j-1])){
          if (getCreateTime(fs,entries[j])>getCreateTime(fs,entries[j-1])){
            break;
          }
        }
        int temp = entries[j];
        entries[j] = entries[j-1];
        entries[j-1] = temp;
        j-=1;
        if (j<1){
          break;
        }
      }
    }
  printf("===sort by file size===\n");
  for (int i = 0; i < len; i++) {
			printf("%s %d\n", getName(fs,entries[i]),getSize(fs,entries[i]));
		}
  }
  else{
    printf("Unknown operation!\n");
    return;
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  gtime+=1;
  if (op==RM){
    int fp = getNameInd(fs,s);
    if (fp==-1){
      printf("%s doesn't exist.\n",s);
      return;
    }
    u32 add = getAdd(fs,fp);
    u32 size = getSize(fs,fp);
    // update FCB valid bit
    fs->volume[fs->SUPERBLOCK_SIZE+fp*fs->FCB_SIZE+31] = 0;
    // clean super block
    int startBlock = add/32;
    int blockNum = size/32;
    if (size%32!=0){
      blockNum += 1;
    }
    for (int i=0;i<blockNum;i++){
      setSuper(fs,startBlock+i,0);
    }
    // compact
    compaction(fs);
  }
  else{
    printf("Unknown operation!\n");
    return;
  }
}
