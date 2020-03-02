/**
 *=============================================================================
 * Project: Camera Writing
 * Project Short Description: Save video to file with input
 * Module: VIDEO_SAVER implementation
 * Author: Trung Nguyen
 * Date: 10/17/2018
 * Viettel Aerospace Institude - Viettel Group
 * ============================================================================
*/


#ifndef GSTSAVER_HPP
#define GSTSAVER_HPP

/** ///////////////////////////////////////////////////////////////////
 *  //  Include Libs
 *  //
 */

//=============== Including preloading C++ Libraries ===== //
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <string.h>

//============== Inluding Gstreamer-1.0 Libraries ======//
#include <glib.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>


//============== Including Opencv Libraries ===========//
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//============== Including User Custom Libs ===========//
#include "gstutility.hpp"
//======== Define function for specific platform ======//
#ifdef __linux__
    //linux code goes here

#elif _WIN32
    // windows code goes here
    #define strcasecmp stricmp
#else

#endif
/**
 * //
 * // Include Done
 * //////////////////////////////////////////////////////////////// */

/** //////////////////////////////
 *  // VideoSaver Declaration
 *  //
 *  //
 */

// Namespace for ProjectName (Product Name)
namespace EyePhoenix {
    struct _GstParams {
        GstElement* mPipeline;
        GMainLoop* mLoop;
        _GstBus *mBus;
        _GstAppSrc *mAppSrc;
        _GstParams()
        {
            mPipeline         = nullptr;
            mLoop             = nullptr;
            mBus              = nullptr;
            mAppSrc           = nullptr;
        }
    };
    struct _VideoConfigs {
        uint32_t mWidth;
        uint32_t mHeight;
        uint32_t mFps;
        _VideoConfigs()
        {
            mWidth = 0;
            mHeight = 0;
            mFps = 0;
        }
        //=== Setter ===//
        void setWidth( const uint32_t& _width ) { mWidth = _width; }
        void setHeight( const uint32_t& _height ) { mHeight = _height; }
        void setFps( const int& _fps ) { mFps = _fps; }

        //=== Getter ===//
        inline uint32_t getWidth() const { return mWidth; }
        inline uint32_t getHeight() const { return mHeight; }
        inline int getFps() const { return mFps; }
    };
    typedef struct _GstParams GstParams;
    typedef struct _VideoConfigs VideoConfigs;

    // Class Decleration
    class VideoSaver {
        // Methods
        public:
            /* =================================================================
              ============================ Public API ==========================
              ==================================================================*/

            /**
             * @brief Create
             * @param encodeType_
             * @param width_
             * @param height_
             * @param fileName_
             * @return
             */
            static VideoSaver* Create( gstEncodeType encodeType_, uint32_t width_, uint32_t height_,
                                uint32_t fps_, char* fileName_ );
            /**
             * @brief open: Set gstreamer pipeline to PLAYING state.
             * @return
             */
            bool open();

            /**
             * @brief close: Close Videosaver pipeline.
             * @return
             */
            bool close();

            /**
             * @brief encodeFrame: Add image raw data to file
             * @param buffer_: raw data
             * @param size_: data size
             * @return
             */
            bool encodeFrame( GstBuffer *buffer_, size_t size_ );

            /**
             * @brief Destructor
             * @return
             */
            ~VideoSaver();
        protected:
            /* =================================================================
              ========================= Protected Methods ======================
              ==================================================================*/
            VideoSaver();
            /**
             * @brief gstreamerPipelineCreation: create pipeline string
             * @return
             */
            bool gstreamerPipelineCreation();

            /**
             * @brief init : init pipeline and callbacks
             * @return
             */
            bool init( gstEncodeType type_, uint32_t width, uint32_t height, char* fileName_ );

            /**
             * @brief onNeedData: Need callback to push data to appsrc
             * @param appsrc
             * @param unused_size
             * @param user_data
             */
            static void onNeedData( GstAppSrc *appsrc, guint unused_size, gpointer user_data );

            /**
             * @brief onEnoughData: Enough callback. Appsrc doesn't need data
             * @param appsrc
             * @param user_data
             */
            static void onEnoughData( GstAppSrc *appsrc, gpointer user_data );


            /* =================================================================
              ========================= Supported Methods ======================
              ==================================================================*/
            /**
             * @brief fileExtension
             * @param fileName
             * @return
             */
            std::string fileExtension( const std::string& fileName );

        //Properties
        private:
            int countFrame = 0;
            gstEncodeType mEncodeType;
            std::string mPipelineStr;
            std::string mCapsAppsrcStr;
            std::string mFileName;
            bool mNeedDataFlag;
            VideoConfigs mVideoConfigs;
            GstParams mGstParams;
    };
};

#endif // GSTSAVER_HPP
