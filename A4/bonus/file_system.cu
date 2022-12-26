#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ u32 getAdd(FileSystem *fs,int ind);  // get the address the file point to
__device__ void storeAdd(FileSystem *fs,int ind,u32 add); // store the address the file point to
__device__ u32 getSize(FileSystem *fs,int ind); // get file size
__device__ void storeSize(FileSystem *fs,int ind,u32 size); // store file size
__device__ u32 getCreateTime(FileSystem *fs,int ind); // get create time
__device__ void storeCreateTime(FileSystem *fs,int ind,u32 time); // store create time
__device__ u32 getModifiedTime(FileSystem *fs,int ind); // get modified time
__device__ void storeModifiedTime(FileSystem *fs,int ind,u32 time); // store modified time
__device__ int getNameInd(FileSystem *fs,char *s); // get the file index in FCB
__device__ char* getName(FileSystem *fs,int ind); // get the file/dir name
__device__ void storeName(FileSystem *fs,int ind,char *s);  // store the file/dir name

__device__ int getEmptyBlock(FileSystem *fs); // function to get empty block
__device__ int getEmptyFCB(FileSystem *fs); // function to get empty FCB
__device__ int getValid(FileSystem *fs,int ind);// check if the file is valid
__device__ void setValid(FileSystem *fs,int ind,int target);// set if the file is valid
__device__ int getIdentity(FileSystem *fs,int ind);// check file or directory
__device__ void setIdentity(FileSystem *fs,int ind,int target);//set file or directory
__device__ int getCurrent(FileSystem *fs,int ind);  // check if in this directory 0:not in 1:in
__device__ void setCurrent(FileSystem *fs,int ind,int target);  // set if in this directory 0:not in 1:in
__device__ void setSuper(FileSystem *fs,int blockInd,int target); // set number of super block
__device__ int getSuper(FileSystem *fs,int blockInd); // get number of super block
__device__ void moveBlock(FileSystem *fs, int originInd, int targetInd); // move the block from one place to another
__device__ void compaction(FileSystem *fs); // function to compact the file system
__device__ int getCurrentDir(FileSystem*fs);// function to get which directory the file system is currently located
__device__ int getParentValid(FileSystem *fs,int ind);// to check if the file/dir has parent
__device__ void setParentValid(FileSystem *fs,int ind,int target); // set if the file/dir has parent
__device__ int getParent(FileSystem *fs,int ind);// get the parent
__device__ void storeParent(FileSystem *fs,int ind,int fp);// store parent
__device__ int getNextValid(FileSystem *fs,int ind);// to check if the file/dir has next
__device__ void setNextValid(FileSystem *fs,int ind,int target); // set if the file/dir has next
__device__ int getNext(FileSystem *fs,int ind);// get the next
__device__ void storeNext(FileSystem *fs,int ind,int fp);// store next
__device__ int getChildValid(FileSystem *fs,int ind);// to check if the directory has child
__device__ void setChildValid(FileSystem *fs,int ind,int target); // set if the directory has child
__device__ int getChild(FileSystem *fs,int ind);// get the first child
__device__ void storeChild(FileSystem *fs,int ind,int fp);// store first child
__device__ int getDirSize(FileSystem *fs,int ind); // get the size of directory
__device__ int isInDir(FileSystem *fs,int current,int fp);//check if the file/dir is in the directory


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
  // init root directory
  setValid(fs,0,1);
  setIdentity(fs,0,1);
  setCurrent(fs,0,1);
  storeCreateTime(fs,0,gtime);
  storeModifiedTime(fs,0,gtime);
  storeName(fs,0,"root\0");
}
// helper function to get and store information of FCB for file

// The file FCB information contains:
// file address: 26-28 address from 0 - 2**23 so 23 bits is enough
// file size: 24-25 --> max file size is 2**10 bits, needs 10 bits, so 16 bits can handle
// time: 20-24 --> create: 20-21; modified: 22-23
// parent pointer: 11 bits
// next pointer: 11 bits
// valid bit: 1
// indentity bit to check if it's file or dir: 1
// in total 24 bits, 3 bytes, 29-31

// The directory FCB information contains:
// time: 20-24 --> create: 20-21; modified: 22-23
// parent pointer: 11 bits
// first child pointer: 11 bits
// next pointer: 11 bits
// a bit to check whether is in the current directory
// in total 36 bits, takes 5 bytes, 26 - 30
// valid bit and indentity bit in 31


//TODO: update get add using bit ops
__device__ int getCurrentDir(FileSystem*fs){
  for (int i=0;i<fs->FCB_ENTRIES;i++){
    if (getCurrent(fs,i)==1){
      return i;
    }
  }
}
__device__ int getParentValid(FileSystem *fs,int ind){
  return (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31]>>2)&0x01;
}
__device__ void setParentValid(FileSystem *fs,int ind,int target){
  if (target == 0){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] &= ~(0x01<<2);
  }
  if (target == 1){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] |= (0x01<<2);
  }
}
__device__ int getParent(FileSystem *fs,int ind){
  int result = 0;
  result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31]&0xf0)>>4;
  result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+30]&0x3f)<<4;
  return result;
}
__device__ void storeParent(FileSystem *fs,int ind,int fp){
  for (int i=0;i<4;i++){
    int temp = (fp>>i)&0x01;
    if (temp==0){
      fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] &= ~(0x01<<(4+i));
    }
    if (temp==1){
      fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] |= 0x01<<(4+i);
    }
  }
  for (int i=0;i<6;i++){
    int temp = (fp>>(4+i))&0x01;
    if (temp==0){
      fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+30] &= ~(0x01<<i);
    }
    if (temp==1){
      fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+30] |= 0x01<<i;
    }
  }
}
__device__ int getNextValid(FileSystem *fs,int ind){
  return (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31]>>3)&0x01;
}
__device__ void setNextValid(FileSystem *fs,int ind,int target){
  if (target == 0){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] &= ~(0x01<<3);
  }
  if (target == 1){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] |= (0x01<<3);
  }
}
__device__ int getNext(FileSystem *fs,int ind){
  int result = 0;
  result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+30]&0xc0)>>6;
  result += fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+29]<<2;
  return result;
}
__device__ void storeNext(FileSystem *fs,int ind,int fp){
  for (int i=0;i<2;i++){
    int temp = (fp>>i)&0x01;
    if (temp==0){
      fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+30] &= ~(0x01<<(6+i));
    }
    if (temp==1){
      fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+30] |= 0x01<<(6+i);
    }
  }
  fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+29] = (fp>>2);
}
__device__ int getChild(FileSystem *fs,int ind){
  int result = 0;
  for (int i=0;i<2;i++){
    result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+26+i]<<(8*i));
  }
  return result;
}
__device__ void storeChild(FileSystem *fs,int ind,int fp){
  for (int i=0;i<2;i++){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+26+i] = fp>>(8*i);
  }
}
__device__ u32 getAdd(FileSystem *fs,int ind){
  u32 result = 0;
  for (int i=0;i<3;i++){
    result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+26+i]<<(8*i));
  }
  return result;
}
__device__ void storeAdd(FileSystem *fs,int ind,u32 add){
  for (int i=0;i<3;i++){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+26+i] = add>>(8*i);
  }
}
__device__ int getDirSize(FileSystem *fs,int ind){
  if (getChildValid(fs,ind)==0){
    return 0;
  }
  int result = 0;
  int temp = getChild(fs,ind);
  char* s = getName(fs,ind);
  while(1){
    char *t = getName(fs,temp);
    int len=0;
    while(1){
      result+=1;
      if (t[len++]=='\0'){
        break;
      }
    }

    if (getNextValid(fs,temp)==0){
      break;
    }
    temp = getNext(fs,temp);
  }
  return result;
}
__device__ u32 getSize(FileSystem *fs,int ind){
  u32 result = 0;
  for (int i=0;i<2;i++){
    result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+24+i]<<(8*i));
  }
  return result;
}
__device__ void storeSize(FileSystem *fs,int ind,u32 size){
  for (int i=0;i<2;i++){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+24+i] = size>>(8*i);
  }
}
__device__ u32 getCreateTime(FileSystem *fs,int ind){
  u32 result = 0;
  for (int i=0;i<2;i++){
    result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+20+i]<<(8*i));
  }
  return result;
}
__device__ u32 getModifiedTime(FileSystem *fs,int ind){
  u32 result = 0;
  for (int i=0;i<2;i++){
    result += (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+22+i]<<(8*i));
  }
  return result;
}
__device__ void storeCreateTime(FileSystem *fs,int ind,u32 time){
  for (int i=0;i<2;i++){
    u32 temp = (time>>(8*i)) & 0xff;
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+20+i] = temp;
  }
}
__device__ void storeModifiedTime(FileSystem *fs,int ind,u32 time){
  for (int i=0;i<2;i++){
    u32 temp = (time>>(8*i)) & 0xff;
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+22+i] = temp;
  }
}
__device__ int isInDir(FileSystem *fs,int current,int fp){
  if (getChildValid(fs,current)==0){
    return 0;
  }
  int temp = getChild(fs,current);
  while(1){
    if (temp == fp){
      return 1;
    }
    if (getNextValid(fs,temp)==0){
      break;
    }
    temp = getNext(fs,temp);
  }
  return 0;
}
__device__ int getNameInd(FileSystem *fs,char *s){
    int ind = -1;
    int superSize = fs->SUPERBLOCK_SIZE;
    int currentDir = getCurrentDir(fs);
    for (int i=0;i<fs->FCB_ENTRIES;i++){
      if (getValid(fs,i)==0){
        continue;
      }
      for (int j=0;j<fs->MAX_FILENAME_SIZE;j++){
        if (fs->volume[superSize+i*fs->FCB_SIZE+j] != s[j]){
          break;
        }
        if (s[j]=='\0' &&(isInDir(fs,currentDir,i)==1)){
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
}

__device__ int getEmptyFCB(FileSystem *fs){
  for (int i = 0;i<fs->FCB_ENTRIES;i++){
    if (getValid(fs,i) == 0){
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
__device__ int getValid(FileSystem *fs,int ind){
  return fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31]&0x01;
}
__device__ void setValid(FileSystem *fs,int ind,int target){
  if (target == 0){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] &= ~(0x01);
  }
  if (target == 1){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] |= 0x01;
  }
}
__device__ int getIdentity(FileSystem *fs,int ind){
  return (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31]>>1)&0x01;
}
__device__ void setIdentity(FileSystem *fs,int ind,int target){
  if (target == 0){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] &= ~(0x01<<1);
  }
  if (target == 1){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+31] |= (0x01<<1);
  }
}
__device__ int getCurrent(FileSystem *fs,int ind){
  return fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+28] & 0x01;
}
__device__ void setCurrent(FileSystem *fs,int ind,int target){
  if (target == 0){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+28] &= ~(0x01);
  }
  if (target == 1){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+28] |= 0x01;
  }
}
__device__ int getChildValid(FileSystem *fs,int ind){
  return (fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+28]>>1) & 0x01;
}
__device__ void setChildValid(FileSystem *fs,int ind,int target){
  if (target == 0){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+28] &= ~(0x01<<1);
  }
  if (target == 1){
    fs->volume[fs->SUPERBLOCK_SIZE+ind*fs->FCB_SIZE+28] |= (0x01<<1);
  }
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
    if (getValid(fs,i)==0 || getIdentity(fs,i)==1){
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
  int currentDir = getCurrentDir(fs);
  int ind = getNameInd(fs,s);
  if (getIdentity(fs,ind)==1){
    printf("This is a directory.\n");
    return -1;
  }
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
        setValid(fs,emptyFCB,1);
        setParentValid(fs,emptyFCB,1);
        storeParent(fs,emptyFCB,currentDir);
        setNextValid(fs,emptyFCB,0);
        setIdentity(fs,emptyFCB,0);
        storeName(fs,emptyFCB,s);
        storeAdd(fs,emptyFCB,emptyBlock*fs->STORAGE_BLOCK_SIZE);
        storeSize(fs,emptyFCB,0);
        storeCreateTime(fs,emptyFCB,gtime);
        storeModifiedTime(fs,emptyFCB,gtime);
        storeModifiedTime(fs,currentDir,gtime);
        // update pointer
        if (getChildValid(fs,currentDir)==0){
          setChildValid(fs,currentDir,1);
          storeChild(fs,currentDir,emptyFCB);
        }
        else{
          int temp = getChild(fs,currentDir);
          while(getNextValid(fs,temp)!=0){
            temp = getNext(fs,temp);
          }
          setNextValid(fs,temp,1);
          storeNext(fs,temp,emptyFCB);
        }
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
  if (getValid(fs,fp)==1){
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
  int currentDir = getCurrentDir(fs);
  int entries[1024];
  int len = 0;
  for (int i=0;i<fs->FCB_ENTRIES;i++){
    if (getValid(fs,i)==1&&(isInDir(fs,currentDir,i)==1)){
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
      if (getIdentity(fs,entries[i])==0){
			  printf("%s\n", getName(fs,entries[i]));
      }
      if (getIdentity(fs,entries[i])==1){
        printf("%s d\n", getName(fs,entries[i]));
      }
		}
  }
  else if (op == LS_S){
    for (int i=1;i<len;i++){
      int j = i;
      int sizeJ;
      int sizeBeforeJ;
      if (getIdentity(fs,entries[j])==0){
        sizeJ = getSize(fs,entries[j]);
      }
      else{
        sizeJ = getDirSize(fs,entries[j]);
      }
      if (getIdentity(fs,entries[j-1])==0){
        sizeBeforeJ = getSize(fs,entries[j-1]);
      }
      else{
        sizeBeforeJ = getDirSize(fs,entries[j-1]);
      }
      while(sizeJ>=sizeBeforeJ){
        if (sizeJ==sizeBeforeJ){
          if (getCreateTime(fs,entries[j])>getCreateTime(fs,entries[j-1])){
            break;
          }
        }
        int temp = entries[j];
        entries[j] = entries[j-1];
        entries[j-1] = temp;
        j-=1;
        if (getIdentity(fs,entries[j-1])==0){
          sizeBeforeJ = getSize(fs,entries[j-1]);
        }
        else{
          sizeBeforeJ = getDirSize(fs,entries[j-1]);
        }
        if (j<1){
          break;
        }
      }
    }
  printf("===sort by file size===\n");
  for (int i = 0; i < len; i++) {
    if (getIdentity(fs,entries[i])==0){
			printf("%s %d\n", getName(fs,entries[i]),getSize(fs,entries[i]));
    }
    if (getIdentity(fs,entries[i])==1){
         printf("%s ",getName(fs,entries[i]));
         printf("%d d\n",getDirSize(fs,entries[i]));
      }
		}
    
  }
  else if (op==CD_P){
    if (getParentValid(fs,currentDir)==0){
      printf("No parent directory.\n");
      return;
    }
    int parent = getParent(fs,currentDir);
    setCurrent(fs,currentDir,0);
    setCurrent(fs,parent,1);
  }
  else if (op==PWD){
    if (getParentValid(fs,currentDir)==0){
      printf("/\n");
      return;
    }
    int path[1024];
    int len = 0;
    int temp = currentDir;
    while(getParentValid(fs,temp)!=0){
      path[len] = temp;
      len+=1;
      temp = getParent(fs,temp);
    }
    for (int i=len-1;i>=0;i--){
      printf("/%s",getName(fs,path[i]));
      if (i==0){
        printf("\n");
      }
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
  int currentDir = getCurrentDir(fs);
  if (op==RM){
    int fp = getNameInd(fs,s);
    if (fp==-1){
      printf("%s doesn't exist.\n",s);
      return;
    }
    if (getIdentity(fs,fp)==1){
      printf("Use RM_RF to remove directory.\n");
      return;
    }
    u32 add = getAdd(fs,fp);
    u32 size = getSize(fs,fp);
    // update FCB valid bit
    setValid(fs,fp,0);
    // clean super block
    int startBlock = add/32;
    int blockNum = size/32;
    if (size%32!=0){
      blockNum += 1;
    }
    for (int i=0;i<blockNum;i++){
      setSuper(fs,startBlock+i,0);
    }
    // update pointer
    if (getParentValid(fs,fp)==1){
      int parent = getParent(fs,fp);
      if (getChild(fs,parent)==fp){
        if (getNextValid(fs,fp)==1){
          storeChild(fs,parent,getNext(fs,fp));
        }
        else{
          setChildValid(fs,parent,0);
        }
      }
      else{
        int temp = getChild(fs,parent);
        while(temp!=fp){
          if(getNext(fs,temp)==fp){
            if (getNextValid(fs,fp)==1){
              storeNext(fs,temp,getNext(fs,fp));
            }
            else{
              setNextValid(fs,temp,0);
            }
            break;
          }
          temp = getNext(fs,temp);
        }
      }
    }
    // compact
    compaction(fs);
    // update modified time of current direcotry
    storeModifiedTime(fs,currentDir,gtime);
  }
  else if (op==RM_RF){
    int fp = getNameInd(fs,s);
    if (fp==-1){
      printf("%s doesn't exist.\n",s);
      return;
    }
    if (getIdentity(fs,fp)==0){
      printf("Use RM to remove file.\n");
      return;
    }
    // update valid bit
    setValid(fs,fp,0);
    // update pointer
    if (getParentValid(fs,fp)==1){
      int parent = getParent(fs,fp);
      if (getChild(fs,parent)==fp){
        if (getNextValid(fs,fp)==1){
          storeChild(fs,parent,getNext(fs,fp));
        }
        else{
          setChildValid(fs,parent,0);
        }
      }
      else{
        int temp = getChild(fs,parent);
        while(1){
          if(getNext(fs,temp)==fp){
            if (getNextValid(fs,fp)==1){
              storeNext(fs,temp,getNext(fs,fp));
            }
            else{
              setNextValid(fs,temp,0);
            }
            break;
          }
          temp = getNext(fs,temp);
        }
      }
    }
    int files[50];
    files[0] = fp;
    int len = 1;
    for (int i=0;i<3;i++){
      if (len == 0){
        break;
      }
      int nextFiles[50];
      int nextLen = 0;
      for (int j=0;j<len;j++){
        if (getChildValid(fs,files[j])==0){
          continue;
        }
        int temp = getChild(fs,files[j]);
        while(1){
          // update FCB valid bit
          setValid(fs,temp,0);
          if (getIdentity(fs,temp)==1){
            nextFiles[nextLen++] = temp;
          }
          if (getIdentity(fs,temp)==0){
            //TODO
            u32 add = getAdd(fs,temp);
            u32 size = getSize(fs,temp);
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
          if (getNextValid(fs,temp)==0){
            break;
          }
          temp = getNext(fs,temp);
        }
      }
      for (int k=0;k<nextLen;k++){
        files[k] = nextFiles[k];
      }
      len = nextLen;
    }
    // update modified time of current direcotry
    storeModifiedTime(fs,currentDir,gtime);
  }
  else if (op==MKDIR){
    int fp = getEmptyFCB(fs);
    if (fp==-1){
       printf("File number over 1024, can not open new file!\n");
      return;
    }
    // update pointer
    if (getChildValid(fs,currentDir)==0){
      setChildValid(fs,currentDir,1);
      storeChild(fs,currentDir,fp);
    }
    else{
      int temp = getChild(fs,currentDir);
      while(getNextValid(fs,temp)!=0){
        temp = getNext(fs,temp);
      }
      setNextValid(fs,temp,1);
      storeNext(fs,temp,fp);
    }
    setValid(fs,fp,1);
    setIdentity(fs,fp,1);
    setParentValid(fs,fp,1);
    storeParent(fs,fp,currentDir);
    setNextValid(fs,fp,0);
    setChildValid(fs,fp,0);
    storeName(fs,fp,s);
    storeCreateTime(fs,fp,gtime);
    storeModifiedTime(fs,fp,gtime);
    storeModifiedTime(fs,currentDir,gtime);
  }
  else if (op==CD){
    int fp = getNameInd(fs,s);
    if (fp==-1){
      printf("%s doesn't exist.\n",s);
      return;
    }
    if (getIdentity(fs,fp)==0){
      printf("Can not cd a file.\n");
      return;
    }
    setCurrent(fs,currentDir,0);
    setCurrent(fs,fp,1);
  }
  else{
    printf("Unknown operation!\n");
    return;
  }
}