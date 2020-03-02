/**
 *=============================================================================
 * Project: Eyephoenix CSI Camera Writing
 * Project Short Description: Save video to file with input
 * Module: VIDEO_SAVER implementation
 * Author: Trung Nguyen
 * Date: 08/06/2018
 * Viettel Aerospace Institude - Viettel Group
 * ============================================================================
*/

#include <sstream>
#include "gstsaver.hpp"

#define LOG_GSTSAVER_FILE "[gstsaver.cpp]"
namespace EyePhoenix {
    // Public Methods

    /* Create instance as factory. */
    VideoSaver* VideoSaver::Create( gstEncodeType encodeType_, uint32_t width_, uint32_t height_, uint32_t fps_, char *fileName_ )
    {
        if( !gstreamerInit() )
        {
            printf(LOG_GSTSAVER_FILE " failed to init gstreamer API\n");
            return nullptr;
        }
        VideoSaver* vidSave = new VideoSaver();
        if( !vidSave )
        {
            return nullptr;
        }
        vidSave->mEncodeType = encodeType_;
        vidSave->mVideoConfigs.setWidth( width_ );
        vidSave->mVideoConfigs.setHeight( height_ );
        vidSave->mVideoConfigs.setFps( fps_ );
        vidSave->mFileName = std::string( fileName_ );
        vidSave->mNeedDataFlag = false;
        if( !vidSave->init( encodeType_, width_, height_, fileName_ ) )
        {
            printf( LOG_GSTSAVER_FILE " failed to init video saving pipeline at line %d - %s", __LINE__, __FILE__ );
            return nullptr;
        }
        return vidSave;
    }


    /* Open video saving pipeline. */
    bool VideoSaver::open()
    {
        // Pipeline transition to PLAYING
        const GstStateChangeReturn result = gst_element_set_state(
                                            GST_ELEMENT( mGstParams.mPipeline ), GST_STATE_PLAYING );
        if( result == GST_STATE_CHANGE_FAILURE )
        {
            printf( LOG_GSTSAVER_FILE " gstreamer video saving failed to transition state to PLAYING at line %d -  %s", __LINE__, __FILE__ );
            return false;
        }
        return true;
    }

    /* Close video saving pipeline. */
    bool VideoSaver::close()
    {
        // send EOS
         mNeedDataFlag = false;
         printf( LOG_GSTSAVER_FILE " gstEncoder - shutting down pipeline, sending EOS\n" );
         GstFlowReturn eos_result = gst_app_src_end_of_stream( mGstParams.mAppSrc );

         if( eos_result != 0 )
             printf( LOG_GSTSAVER_FILE " gstEncoder - failed sending appsrc EOS (result %u)\n", eos_result );

         // stop pipeline
         printf( LOG_GSTSAVER_FILE " gstEncoder - transitioning pipeline to GST_STATE_NULL\n" );

         const GstStateChangeReturn result = gst_element_set_state( mGstParams.mPipeline, GST_STATE_NULL);

         if( result != GST_STATE_CHANGE_SUCCESS )
             printf( LOG_GSTSAVER_FILE " gstEncoder - failed to set pipeline state to NULL (error %u)\n", result );

         printf( LOG_GSTSAVER_FILE " gstEncoder - pipeline shutdown complete\n" );
         return true;
    }

    /* Add frame to saving pipeline. */

    bool VideoSaver::encodeFrame( GstBuffer *buffer_, size_t size_ = 1920*1080 )
    {
        if( !buffer_ || size_ == 0 )
        {
            return false;
        }

        if( !mNeedDataFlag )
        {
            printf( LOG_GSTSAVER_FILE " Pipeline full, skipping frame. \n" );
            return true;
        }
        // queue buffer to gstreamer
        GstFlowReturn ret;
        GstClockTime gstDuration = GST_SECOND/mVideoConfigs.getFps();
        GST_BUFFER_PTS(buffer_)=(countFrame+1)*gstDuration;
        GST_BUFFER_DTS(buffer_)=(countFrame+1)*gstDuration;

        GST_BUFFER_DURATION(buffer_)=gstDuration;
        GST_BUFFER_OFFSET(buffer_)=countFrame+1;
        countFrame++;
        g_signal_emit_by_name(mGstParams.mAppSrc, "push-buffer", buffer_, &ret);
//        gst_buffer_unref(buffer_);

        if( ret != 0 )
            printf(LOG_GSTSAVER_FILE "gstEncoder - AppSrc pushed buffer abnormally (result %u)\n", ret);

        // check for any messages
        while(true)
        {
            GstMessage* msg = gst_bus_pop( mGstParams.mBus );

            if( !msg )
                break;

            gst_message_print( mGstParams.mBus, msg, this );
            gst_message_unref( msg );
        }
        return true;
    }

    // Protected Methods
    /* Constructor */
    VideoSaver::VideoSaver()
    {
        mEncodeType = gstEncodeType::GST_CODEC_H264;
        mCapsAppsrcStr = "";
        mNeedDataFlag = false;
    }

    /* Destructor */
    VideoSaver::~VideoSaver()
    {
        try {
            close();
        }catch(...)
        {
            printf("Error in videosaver destructor");
        }
    }
    
    /* Gstreamer pipeline creation */ 
    bool VideoSaver::gstreamerPipelineCreation()
    {
        std::ostringstream ss;
        ss << "appsrc name=mySrc ! video/x-raw,width=" << mVideoConfigs.getWidth() << ",height=" << mVideoConfigs.getHeight();
        ss << ",framerate=" << mVideoConfigs.getFps() << "/1,format=I420 ! ";
        if( mEncodeType == gstEncodeType::GST_CODEC_H264 )
        {
            ss << "x264enc bitrate=4096 tune=zerolatency ! video/x-h264,stream-format=byte-stream ! h264parse ! "  /*quality-level=2 */;
        }
        else if ( mEncodeType == gstEncodeType::GST_CODEC_H265)
        {
            ss << "x265enc bitrate=8192 tune=zerolatency ! video/x-h265,stream-format=byte-stream ! h265parse ! " /*quality-level=2*/;
        }
        else if( mEncodeType == gstEncodeType::GST_CODEC_MPEG)
        {
            ss << "avenc_mpeg4 bitrate=800000 max-threads=4 max-bframes=10 ! video/mpeg ! ";
        }
        else
        {
            printf(LOG_GSTSAVER_FILE " CODEC is not supported. \n");
            return false;
        }
        std::string ext = fileExtension( mFileName );
        if( strcasecmp( ext.c_str(), "mkv") == 0 )
        {
            ss << "matroskamux ! ";
        }
        else if( strcasecmp( ext.c_str(), "mp4") == 0 )
        {
            ss << "mpegtsmux ! ";
        }
        else if( strcasecmp( ext.c_str(), "avi") == 0 )
        {
            ss << "avimux ! ";
        }
        else if( strcasecmp( ext.c_str(), "h264") && strcasecmp( ext.c_str(), "h265" ) )
        {
            printf( LOG_GSTSAVER_FILE " invalid output extension - %s\n", ext.c_str() );
            return false;
        }
        ss << "filesink location=" << mFileName;
        mPipelineStr = ss.str();
        printf( LOG_GSTSAVER_FILE " gst pipeline: %s", mPipelineStr.c_str() );
        return true;
    }

    /* Init pipeline and callbacks */
    bool VideoSaver::init( gstEncodeType type_,
                           uint32_t width_, uint32_t height_, char *fileName_ )
    {
        mEncodeType = type_;
        mVideoConfigs.setWidth( width_ );
        mVideoConfigs.setHeight( height_ );
        mFileName = std::string( fileName_ );
        if( ! gstreamerPipelineCreation() )
        {
            printf(LOG_GSTSAVER_FILE " failed to create pipeline string. \n");
            return false;
        }

        GError* err  = NULL;
        mGstParams.mPipeline = gst_parse_launch( mPipelineStr.c_str(), &err );
        if( err != NULL )
        {
            printf( LOG_GSTSAVER_FILE "gstreamer video saving failed to create pipeline at %d - %s\n", __LINE__, __FILE__ );
            printf( LOG_GSTSAVER_FILE "   (%s)\n", err->message);
            g_error_free(err);
            return false;
        }

        GstPipeline* pipeline  = GST_PIPELINE( mGstParams.mPipeline );
        if( !pipeline )
        {
            printf(LOG_GSTSAVER_FILE "gstreamer video saving failed to cast GstElement into GstPipeline at line %d - %s \n", __LINE__, __FILE__ );
            return false;
        }

        mGstParams.mBus = gst_pipeline_get_bus( pipeline );
        if( !mGstParams.mBus )
        {
            printf( LOG_GSTSAVER_FILE " gstreamer video saving failed to retreived bus messages at line %d - %s", __LINE__, __FILE__  );
            return false;
        }
        // Get the appsrc
        GstElement* appsrcElement = gst_bin_get_by_name( GST_BIN( mGstParams.mPipeline), "mySrc" );
        mGstParams.mAppSrc = GST_APP_SRC( appsrcElement );
        g_object_set( G_OBJECT( mGstParams.mAppSrc ),
                         "is-live", TRUE,
                         "num-buffers", -1,
                         "emit-signals", FALSE,
                         "format", GST_FORMAT_TIME,
                         "stream-type", GST_APP_STREAM_TYPE_STREAM,
                         "do-timestamp", TRUE,
                         nullptr );
        printf("Width------------%d-%d\n", width_, height_);
        std::ostringstream ss;
        ss << "video/x-raw,width=" << width_ << ",height=" << height_ << ",format=I420";
        GstCaps *caps =  gst_caps_from_string( ss.str().c_str() );
        gst_app_src_set_caps( mGstParams.mAppSrc, caps );
        gst_caps_unref( caps );
        if( !appsrcElement || !mGstParams.mAppSrc )
        {
            printf( LOG_GSTSAVER_FILE " gstreamer failed to retrieve AppSrc element from pipeline at line %d - %s \n", __LINE__, __FILE__ );
            return false;
        }
        GstAppSrcCallbacks cbs;
        cbs.need_data = onNeedData;
        cbs.enough_data = onEnoughData;
        gst_app_src_set_callbacks(GST_APP_SRC_CAST(mGstParams.mAppSrc), &cbs, (void*) this, NULL);
        //g_signal_connect( appsrcElement, "need-data", G_CALLBACK( onNeedData), (void*) this );
        //g_signal_connect( appsrcElement, "enough-data", G_CALLBACK( onEnoughData), (void* ) this );
        return true;
    }
    /* onNeedData event callback*/
    void VideoSaver::onNeedData( GstAppSrc *appsrc, guint unused_size, gpointer user_data )
    {
//        printf( LOG_GSTSAVER_FILE " Appsrc requesting data (%u bytes)\n", unused_size );
        if( !user_data )
            return;
        VideoSaver* vid = ( VideoSaver* ) user_data;
        vid->mNeedDataFlag = true;
    }

    /* onEnoughData event callback*/
    void VideoSaver::onEnoughData( GstAppSrc *appsrc, gpointer user_data )
    {
        printf( LOG_GSTSAVER_FILE " Appsrc signalling enough data\n" );
        if( !user_data )
            return;
        VideoSaver* vid =  ( VideoSaver* ) user_data;
        vid->mNeedDataFlag = false;
    }

    /* FileExtension checking supported method */
    std::string VideoSaver::fileExtension( const std::string& fileName )
    {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream( fileName );
        char delimiter = '.';
        while ( std::getline(tokenStream, token, delimiter) )
        {
           tokens.push_back(token);
        }
        return tokens.back();
    }
};

