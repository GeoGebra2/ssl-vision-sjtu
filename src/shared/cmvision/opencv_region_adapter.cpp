// OpenCV adapter implementation
#include "opencv_region_adapter.h"
#include <opencv2/opencv.hpp>
#include "image.h"

namespace CMVision {

void buildRegionListFromThreshold(const Image<raw8> * threshold_map, CMVision::RegionList * reglist, CMVision::ColorRegionList * colorlist, int min_blob_area, double /*min_pixel_ratio*/) {
  if (!threshold_map || !reglist || !colorlist) return;
  int width = threshold_map->getWidth();
  int height = threshold_map->getHeight();

  // reset color lists
  int num_colors = colorlist->getNumColorRegions();
  CMVision::RegionLinkedList * colorArr = colorlist->getColorRegionArrayPointer();
  for (int i = 0; i < num_colors; ++i) colorArr[i].reset();

  CMVision::Region * regions = reglist->getRegionArrayPointer();
  int max_regions = reglist->getMaxRegions();
  int used = 0;

  // wrap threshold_map data into cv::Mat (no copy)
  cv::Mat full_mat(height, width, CV_8UC1, (void*)threshold_map->getData());

  int global_max_area = 0;

  for (int c = 0; c < num_colors; ++c) {
    cv::Mat bin;
    cv::compare(full_mat, (uchar)c, bin, cv::CMP_EQ); // 0 or 255

    // connected components
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
    for (int lbl = 1; lbl < n; ++lbl) {
      int area = stats.at<int>(lbl, cv::CC_STAT_AREA);
      if (area < min_blob_area) continue;
      if (used >= max_regions) break;

      int left = stats.at<int>(lbl, cv::CC_STAT_LEFT);
      int top  = stats.at<int>(lbl, cv::CC_STAT_TOP);
      int w    = stats.at<int>(lbl, cv::CC_STAT_WIDTH);
      int h    = stats.at<int>(lbl, cv::CC_STAT_HEIGHT);
      double cx = centroids.at<double>(lbl, 0);
      double cy = centroids.at<double>(lbl, 1);

      CMVision::Region & r = regions[used];
      r.color = (raw8)c;
      r.x1 = left; r.y1 = top; r.x2 = left + w - 1; r.y2 = top + h - 1;
      r.cen_x = (float)cx; r.cen_y = (float)cy;
      r.area = area;
      r.run_start = 0;
      r.iterator_id = 0;
      r.next = 0;
      r.tree_next = 0;

      // insert front into color linked list
      colorArr[c].insertFront(&r);

      used++;
      if (area > global_max_area) global_max_area = area;
    }
    if (used >= max_regions) break;
  }

  reglist->setUsedRegions(used);

  // sort regions by area (reuse existing function)
  CMVision::RegionProcessing::sortRegions(colorlist, global_max_area);
}

}
