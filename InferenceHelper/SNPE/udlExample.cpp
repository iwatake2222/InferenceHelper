//==============================================================================
//
//  Copyright (c) 2016-2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <iterator>
#include <vector>
#include <string>
#include <numeric>

#include "udlExample.hpp"

#define DEADBEAF_PTR (void*)(static_cast<intptr_t>(0xdeadbeaf))

namespace {

std::size_t getSizeByDim(const std::vector<size_t>& dim) {
   return std::accumulate(std::begin(dim),
                          std::end(dim),
                          1,
                          std::multiplies<size_t>());
}

void printArray(const size_t* start, const size_t* end) {
   (void)std::for_each(start, end, [](const size_t i){
         std::cout << i << " ";
      });

}

bool isPassthrough(const std::string& type) {
   std::string lower;
   (void)std::transform(std::begin(type), std::end(type),
                        std::back_inserter(lower), [](const char c){ return std::tolower(c); });
   if (lower != std::string("passthrough")) {
      std::cerr << "isPassthrough expecting type passthrough got " <<
         type << std::endl;
      return false;
   }
   return true;
}

} // ns

namespace UdlExample {

zdl::DlSystem::IUDL* MyUDLFactory(void* cookie, const zdl::DlSystem::UDLContext* c) {
   if (!c) return nullptr;
   if (cookie != DEADBEAF_PTR) {
      std::cerr << "MyUDLFactory cookie should be 0xdeadbeaf" << std::endl;
      return nullptr;
   }

   if (!isPassthrough(c->getType())) {
      std::cerr << "MyUDLFactory expecting Passthrough layer, got " << c->getType() << std::endl;
      return nullptr;
   }
   return new UdlPassthrough(*c);
}

bool UdlPassthrough::setup(void* cookie,
                           size_t insz, const size_t* indim[],  const size_t indimsz[],
                           size_t outsz, const size_t* outdim[], const size_t outdimsz[]) {
   if (cookie != DEADBEAF_PTR) {
      std::cerr << "UdlPassthrough::setup() cookie should be 0xdeadbeaf" << std::endl;
      return false;
   }

   // FIXME we need to use proper logging here not using streams
   std::cout << "UdlPassthrough::setup() of name " << m_Context.getName() <<
      " and of type " << m_Context.getType() << std::endl;

   if (!isPassthrough(m_Context.getType())) {
      std::cerr << "UdlPassthrough::setup() expecting passthrough layer type got " <<
         m_Context.getType() << std::endl;
      return false;
   }

   std::cout << "                        input array size " << insz << std::endl;
   std::cout << "                        output array size " << outsz << std::endl;

   // print the input/output dims
   std::cout << "UdlPassthrough::setup() input dims\n";
   (void)std::copy(indimsz, indimsz + insz, std::ostream_iterator<size_t> (std::cout, " "));
   std::cout << std::endl;
   size_t idx = 0;
   (void)std::for_each(indim, indim + insz, [&idx, &indimsz](const size_t* arr){
         std::cout << "[";
         printArray(arr, arr + indimsz[idx]);
         std::cout << "]";
         ++idx;
      });
   std::cout << std::endl;
   std::cout << "UdlPassthrough::setup() output dims\n";
   (void)std::copy(outdimsz, outdimsz + insz, std::ostream_iterator<size_t> (std::cout, " "));
   std::cout << std::endl;
   idx = 0;
   (void)std::for_each(outdim, outdim + outsz, [&idx, &outdimsz](const size_t* arr){
         std::cout << "[";
         printArray(arr, arr + outdimsz[idx]);
         std::cout << "]";
         ++idx;
      });
   std::cout << std::endl;

   if (insz != outsz) {
      std::cerr << "UdlPassthrough::setup() not the same number of dim, in:" <<
         insz  << " != : " << outsz  << std::endl;
      return false;
   }
   m_Insz = insz;
   size_t cnt = insz;

   // If the user want to refer to the indim[] and outdim[],
   // he/she needs to make a copy of this arrays.
   // After setup, these arrays are destroyes, so you cannot cache it as is
   m_OutSzDim.reserve(cnt);
   while (cnt-- > 0) {
      // compute dims and compare. keep the output dim
      const size_t *indims   = indim[cnt];
      const size_t inszdim   = getSizeByDim(std::vector<size_t>(indims, indims + indimsz[cnt]));
      const size_t *outdims  = outdim[cnt]; // insz == outsz
      m_OutSzDim[cnt] = getSizeByDim(std::vector<size_t>(outdims, outdims + outdimsz[cnt]));

      std::cout << "UdlPassthrough::setup() input size for index " << cnt
                << " is dim: " << inszdim << ", output: " << m_OutSzDim[cnt]  <<std::endl;
      if (inszdim != m_OutSzDim[cnt]) {
         std::cerr << "UdlPassthrough::setup() not the same overall dim, in:" <<
         inszdim  << " != out: " << m_OutSzDim[cnt]  << std::endl;
         return false;
      }
   }
   // parse the Passthrough params
   const uint8_t* blob = m_Context.getBlob();
   std::cout << "UdlPassthrough::setup() got blob size " <<  m_Context.getSize() << std::endl;
   if (!blob) {
      std::cout << "UdlPassthrough::setup() got null blob " << std::endl;
      return false;
   }
   // Python packing is this way:
   // self._blob = struct.pack('I', params.blob_count)
   // 'I' here means 32bit - https://docs.python.org/2/library/struct.html
   std::cout << "UdlPassthrough::setup() got blob content " <<
      *(reinterpret_cast<const int32_t*>(blob)) << std::endl;
   return true;
}

void UdlPassthrough::close(void* cookie) noexcept {
   if (cookie != DEADBEAF_PTR) {
      std::cerr << "UdlPassthrough::close() cookie should be 0xdeadbeaf" << std::endl;
   }
   std::cout << "UdlPassthrough::close()" << std::endl;
   delete this;
}

bool UdlPassthrough::execute(void *cookie, const float **input, float **output) {
   if (cookie != DEADBEAF_PTR) {
      std::cerr << "UdlPassthrough::execute() cookie should be 0xdeadbeaf" << std::endl;
      return false;
   }
   std::cout << "UdlPassthrough::execute() number of I/Os is:" << m_Insz << std::endl;
   // 0...m_OutSzDim --> going backwards
   // m_OutSzDim is assumed to be != 0
   size_t cnt = m_Insz;
   while (cnt-- > 0) {
      std::cout << std::dec;
      std::cout << "UdlPassthrough::execute() running index " << cnt << std::endl;
      size_t dim = sizeof(float) * m_OutSzDim[cnt];
      std::cout << "UdlPassthrough::execute() dims (a*b*c*...) is:" << m_OutSzDim[cnt] << std::endl;
      std::cout << "UdlPassthrough::execute() dim(total number of bytes) is:" << dim << std::endl;

      const float *i = input[cnt];
      float *o = output[cnt];
      if (!i || !o) {
         std::cerr << "Input or output cannot be 0" << std::endl;
         return false;
      }
      std::cout << "input: 0x" << std::hex << std::setw(8) << std::setfill('0') << i << std::endl;
      std::cout << "output: 0x" << std::hex << std::setw(8) << std::setfill('0') << o << std::endl;
      std::memcpy(o, i, dim);
   }
   return true;
}

} // ns batchrun
