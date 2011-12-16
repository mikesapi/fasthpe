#ifndef SM_API_TESTAPPCONSOLE_UTILS_H
#define SM_API_TESTAPPCONSOLE_UTILS_H

#include "lock.h"

#define THROW_ON_ERROR(x) \
{ \
    smReturnCode result = (x); \
    if (result < 0) \
    { \
        std::stringstream s; \
        s << "API error code: " << result; \
        throw std::runtime_error(s.str()); \
    } \
}

namespace ftn
{
    namespace mikesapi
    {
        namespace samplecode
        {
            // Global variables
            Mutex g_mutex;
            bool g_ctrl_c_detected(false);
            bool g_do_head_pose_printing(false);
            bool g_do_face_data_printing(false);
            unsigned short g_overlay_flags(0);

            // CTRL-C handler function
            void __cdecl CtrlCHandler(int)
            {
                Lock lock(g_mutex);
                std::cout << "Ctrl-C detected, stopping..." << std::endl;
                g_ctrl_c_detected = true;
            }

            // Radians to degrees conversion
            float rad2deg(float rad)
            {
                return rad*57.2957795f;
            }

            void toggleFlag(unsigned short &val, unsigned short flag)
            {
                if (val & flag)
                {
                    val = val & ~flag;
                }
                else
                {
                    val = val | flag;
                }
            }

            //// Stream operator for printing head-pose data
            //std::ostream &operator<<(std::ostream & os, const smEngineHeadPoseData &head_pose)
            //{
            //    fixed(os);
            //    showpos(os);
            //    os.precision(2);
            //    return os << "Head Pose: " 
            //              << "head_pos" << head_pose.head_pos << " " 
            //              << "head_rot" << head_pose.head_rot << " "
            //              << "left_eye_pos" << head_pose.left_eye_pos << " "
            //              << "right_eye_pos" << head_pose.right_eye_pos << " " 
            //              << "confidence " << head_pose.confidence;
            //}

            //std::ostream &operator<<(std::ostream & os, const smCameraVideoFrame &vf)
            //{
            //    fixed(os);
            //    showpos(os);
            //    os.precision(2);
            //    return os << "Framenum: " << vf.frame_num;
            //}

            // Handles keyboard events: return false if quit.
            bool processKeyPress()
            {
                Lock lock(g_mutex);
                if (g_ctrl_c_detected)
                {
                    return false;
                }
                if (!_kbhit())
                {
                    return true;
                }
                int key = _getch();
                switch (key)
                {
                case 'q':
                    return false;
                case 'r':
                    {
                        // Manually restart the tracking
//                        THROW_ON_ERROR(smEngineStart(engine_handle));
                        std::cout << "Restarting tracking" << std::endl; 
                    }
                    return true;
                case 'a':
                    {
                        // Toggle auto-restart mode
                        int on;
//                        THROW_ON_ERROR(smHTGetAutoRestartMode(engine_handle,&on));
//                        THROW_ON_ERROR(smHTSetAutoRestartMode(engine_handle,!on));
                        std::cout << "Autorestart-mode is " << (on?"on":"off") << std::endl; 
                    }
                    return true;
                default:
                    return true;
                }
            }

            // Setup console window geometry / font etc
            void initConsole()
            {
                HANDLE console_handle = GetStdHandle(STD_OUTPUT_HANDLE);
                // Buffer of 255 x 1024
                COORD buffer_size;
                buffer_size.X = 255;
                buffer_size.Y = 1024;
                SetConsoleScreenBufferSize(console_handle, buffer_size);
                // Window size of 120 x 50
                SMALL_RECT window_size;
                window_size.Left = 0;
                window_size.Right = 120;
                window_size.Top = 0;
                window_size.Bottom = 50;
                SetConsoleWindowInfo(console_handle,TRUE,&window_size);
                // Green text
                SetConsoleTextAttribute(console_handle, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
//				ShowWindow(GetConsoleWindow(), SW_HIDE);
            }
        }
    }
}

#endif
