// OpenCV-based adapter to build CMVision RegionList from threshold image
#ifndef OPENCV_REGION_ADAPTER_H
#define OPENCV_REGION_ADAPTER_H

#include "cmvision_region.h"

namespace CMVision {

// Build RegionList + ColorRegionList directly from a threshold image
// threshold_map: Image<raw8> where pixel value is color index
void buildRegionListFromThreshold(const Image<raw8> * threshold_map, CMVision::RegionList * reglist, CMVision::ColorRegionList * colorlist, int min_blob_area, double min_pixel_ratio);

}

#endif
