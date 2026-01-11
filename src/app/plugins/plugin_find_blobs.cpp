//========================================================================
//  This software is free: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License Version 3,
//  as published by the Free Software Foundation.
//
//  This software is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  Version 3 in the file COPYING that came with this distribution.
//  If not, see <http://www.gnu.org/licenses/>.
//========================================================================
/*!
  \file    plugin_find_blobs.cpp
  \brief   C++ Implementation: plugin_find_blobs
  \author  Stefan Zickler, 2008
*/
//========================================================================
#include "plugin_find_blobs.h"
#ifdef USE_OPENCV_BLOBS
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#endif

PluginFindBlobs::PluginFindBlobs(FrameBuffer * _buffer, YUVLUT * _lut)
 : VisionPlugin(_buffer)
{
  lut=_lut;

  _settings=new VarList("Blob Finding");
  _settings->addChild(_v_min_blob_area=new VarInt("min_blob_area", 5));
  _settings->addChild(_v_min_blob_area_ratio=new VarDouble("min_blob_area ratio", 0.5));
  _settings->addChild(_v_enable=new VarBool("enable", true));
  _settings->addChild(v_max_regions=new VarInt("max regions", 50000, 10000, 1000000));

}


PluginFindBlobs::~PluginFindBlobs()
{
  delete _settings;
  delete _v_min_blob_area;
  delete _v_min_blob_area_ratio;
  delete _v_enable;
  delete v_max_regions;
}



ProcessResult PluginFindBlobs::process(FrameData * data, RenderOptions * options) {
  (void)options;


  CMVision::RegionList * reglist = (CMVision::RegionList *) data->map.get("cmv_reglist");
  if (reglist == nullptr || reglist->getMaxRegions() != v_max_regions->getInt()) {
    delete reglist;
    reglist = (CMVision::RegionList *) data->map.update("cmv_reglist", new CMVision::RegionList(v_max_regions->getInt()));
  }

  CMVision::ColorRegionList * colorlist = (CMVision::ColorRegionList *) data->map.get("cmv_colorlist");
  if (colorlist == nullptr) {
    colorlist = (CMVision::ColorRegionList *) data->map.insert("cmv_colorlist", new CMVision::ColorRegionList(lut->getChannelCount()));
  }

  CMVision::RunList * runlist = (CMVision::RunList *) data->map.get("cmv_runlist");
  if (runlist == nullptr) {
    printf("Blob finder: no runlength-encoded input list was found!\n");
    return ProcessingFailed;
  }

  if (_v_enable->getBool()) {
#ifdef USE_OPENCV_BLOBS
    // Fast OpenCV connected components per color channel (bypass runlength path)
    Image<raw8> * img_thresholded = (Image<raw8> *) data->map.get("cmv_threshold");
    if (img_thresholded != nullptr) {
      int width = img_thresholded->getWidth();
      int height = img_thresholded->getHeight();
      raw8 * pixels = img_thresholded->getPixelData();

      CMVision::Region * reg = reglist->getRegionArrayPointer();
      CMVision::RegionLinkedList * color = colorlist->getColorRegionArrayPointer();
      int max_reg = reglist->getMaxRegions();
      int num_colors = colorlist->getNumColorRegions();

      // clear color lists
      for (int i=0;i<num_colors;i++) color[i].reset();

      int n = 0;
      int global_max_area = 0;

      for (int c=0;c<num_colors;c++) {
        // build binary image for this color
        cv::Mat bin(height, width, CV_8UC1);
        const int total = width * height;
        for (int i=0;i<total;i++) bin.data[i] = (pixels[i] & (1u<<c)) ? 255 : 0;

        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);

        for (int lbl=1; lbl<num_labels; lbl++) {
          int area = stats.at<int>(lbl, cv::CC_STAT_AREA);
          if (area < _v_min_blob_area->getInt()) continue;

          int left = stats.at<int>(lbl, cv::CC_STAT_LEFT);
          int top = stats.at<int>(lbl, cv::CC_STAT_TOP);
          int w = stats.at<int>(lbl, cv::CC_STAT_WIDTH);
          int h = stats.at<int>(lbl, cv::CC_STAT_HEIGHT);

          if (n >= max_reg) {
            reglist->setUsedRegions(max_reg);
            goto finish_opencv_blobs;
          }

          reg[n].color.v = (uint8_t)c;
          reg[n].area = area;
          reg[n].x1 = left;
          reg[n].y1 = top;
          reg[n].x2 = left + w - 1;
          reg[n].y2 = top + h - 1;
          reg[n].cen_x = (float)centroids.at<double>(lbl,0);
          reg[n].cen_y = (float)centroids.at<double>(lbl,1);
          reg[n].run_start = 0;
          reg[n].iterator_id = 0;
          reg[n].next = 0;

          // insert into color list
          color[c].insertFront(&reg[n]);
          if (area > global_max_area) global_max_area = area;
          n++;
        }
      }
finish_opencv_blobs:
      reglist->setUsedRegions(n);
      CMVision::RegionProcessing::sortRegions(colorlist, global_max_area);
      return ProcessingOk;
    }
#endif

    //Connect the components of the runlength map:
    CMVision::RegionProcessing::connectComponents(runlist);

    //Extract Regions from runlength map:
    CMVision::RegionProcessing::extractRegions(reglist, runlist);

    if (reglist->getUsedRegions() == reglist->getMaxRegions()) {
      printf("Warning: FindBlobs: extract regions exceeded maximum number of %d regions\n",reglist->getMaxRegions());
    }

    //Separate Regions by colors:
    int max_area = CMVision::RegionProcessing::separateRegions(colorlist, reglist, _v_min_blob_area->getInt(), _v_min_blob_area_ratio->getDouble());

    //Sort Regions:
    CMVision::RegionProcessing::sortRegions(colorlist,max_area);
  } else {
    //detect nothing.
    reglist->setUsedRegions(0);
    int num_colors=colorlist->getNumColorRegions();
    CMVision::RegionLinkedList * color=colorlist->getColorRegionArrayPointer();

    // clear out the region list head table
    for(int i=0; i<num_colors; i++){
      color[i].reset();
    }
  }

  return ProcessingOk;

}

VarList * PluginFindBlobs::getSettings() {
  return _settings;
}

string PluginFindBlobs::getName() {
  return "FindBlobs";
}
