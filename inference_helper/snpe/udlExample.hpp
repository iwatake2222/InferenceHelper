//==============================================================================
//
//  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef UDL_EXAMPLE_HPP
#define UDL_EXAMPLE_HPP

// Include IUDL using the prefix
#include "DlSystem/IUDL.hpp"
#include "DlSystem/UDLContext.hpp"

namespace  UdlExample{

zdl::DlSystem::IUDL* MyUDLFactory(void*, const zdl::DlSystem::UDLContext*);

// Example of Passthrough layer that looks like this:
// layer {
//   name: "Passthrough1"
//   type: "Passthrough"
//   bottom: "norm1"
//   top: "norm1"
//   passthrough_param {
//     blob_count: 1
//   }
// }
class UdlPassthrough final : public zdl::DlSystem::IUDL {
public:
   UdlPassthrough(const UdlPassthrough&) = delete;
   UdlPassthrough& operator=(const UdlPassthrough&) = delete;

   /**
    * @brief UDLContext by value but it has move operation
    */
   UdlPassthrough(zdl::DlSystem::UDLContext context) : m_Context(context) {}

   /**
    * @brief Setup User's environment.
    *        This is being called by DnnRunTime framework
    *        to let the user opportunity to setup anything
    *        which is needed for running user defined layers
    * @return true on success, false otherwise
    */
   virtual bool setup(void *cookie,
                      size_t insz, const size_t **indim, const size_t *indimsz,
                      size_t outsz, const size_t **outdim, const size_t *outdimsz) override;

   /**
    * Close the instance. Invoked by DnnRunTime to let
    * the user the opportunity to close handels etc...
    */
   virtual void close(void *cookie) noexcept override;

   /**
    * Execute the user defined layer
    * will contain the return value/output tensor
    */
   virtual bool execute(void *cookie, const float **input, float **output) override;
private:
   zdl::DlSystem::UDLContext m_Context;
   // this is a*b*c*...*n
   std::vector<size_t> m_OutSzDim;
   // cache the insz/outsz of the incoming
   size_t m_Insz = 0;
   // No need for this since in passthrough its all the same
   // size_t m_Outsz = 0;
};

} // ns sample

#endif // UDL_EXAMPLE_HPP
