using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-1)]
public class ChangeCameraParameters: MonoBehaviour {

    void Start()
    {
        GameObject cameras = GameObject.Find("Cameras");
        // Change parameters of CameraToCapture1
        {
            GameObject cameraToCapture1=GameObject.Find("Camera1");

            Vector3 old_pos = cameraToCapture1.transform.position;
            Vector3 new_pos;
            /*CAMERA1_POSITION_X*/ new_pos.x = old_pos.x;
            /*CAMERA1_POSITION_Y*/ new_pos.y = old_pos.y;
            /*CAMERA1_POSITION_Z*/ new_pos.z = old_pos.z;
            cameraToCapture1.transform.position = cameras.transform.TransformPoint(new_pos);

            Vector3 old_angle = cameraToCapture1.transform.eulerAngles;
            Vector3 new_angle;
            /*CAMERA1_ANGLE_X*/ new_angle.x = old_angle.x;
            /*CAMERA1_ANGLE_Y*/ new_angle.y = old_angle.y;
            /*CAMERA1_ANGLE_Z*/ new_angle.z = 0;
            cameraToCapture1.transform.eulerAngles = cameras.transform.TransformDirection(new_angle);

            Camera cam = cameraToCapture1.GetComponent<Camera>();
            float old_focalLength = cam.focalLength;
            float new_focalLength;
            /*CAMERA1_FOCALLENGTH*/ new_focalLength = old_focalLength;
            cam.focalLength = new_focalLength;
        }

        // Change parameters of CameraToCapture2
        {
            GameObject cameraToCapture2=GameObject.Find("Camera2");

            Vector3 old_pos = cameraToCapture2.transform.position;
            Vector3 new_pos;
            /*CAMERA2_POSITION_X*/ new_pos.x = old_pos.x;
            /*CAMERA2_POSITION_Y*/ new_pos.y = old_pos.y;
            /*CAMERA2_POSITION_Z*/ new_pos.z = old_pos.z;
            cameraToCapture2.transform.position = cameras.transform.TransformPoint(new_pos);

            Vector3 old_angle = cameraToCapture2.transform.eulerAngles;
            Vector3 new_angle;
            /*CAMERA2_ANGLE_X*/ new_angle.x = old_angle.x;
            /*CAMERA2_ANGLE_Y*/ new_angle.y = old_angle.y;
            /*CAMERA2_ANGLE_Z*/ new_angle.z = 0;
            cameraToCapture2.transform.eulerAngles = cameras.transform.TransformDirection(new_angle);

            Camera cam = cameraToCapture2.GetComponent<Camera>();
            float old_focalLength = cam.focalLength;
            float new_focalLength;
            /*CAMERA2_FOCALLENGTH*/ new_focalLength = old_focalLength;
            cam.focalLength = new_focalLength;
        }

        // Change parameters of CameraToCapture3
        {
            GameObject cameraToCapture3=GameObject.Find("Camera3");

            Vector3 old_pos = cameraToCapture3.transform.position;
            Vector3 new_pos;
            /*CAMERA3_POSITION_X*/ new_pos.x = old_pos.x;
            /*CAMERA3_POSITION_Y*/ new_pos.y = old_pos.y;
            /*CAMERA3_POSITION_Z*/ new_pos.z = old_pos.z;
            cameraToCapture3.transform.position = cameras.transform.TransformPoint(new_pos);

            Vector3 old_angle = cameraToCapture3.transform.eulerAngles;
            Vector3 new_angle;
            /*CAMERA3_ANGLE_X*/ new_angle.x = old_angle.x;
            /*CAMERA3_ANGLE_Y*/ new_angle.y = old_angle.y;
            /*CAMERA3_ANGLE_Z*/ new_angle.z = 0;
            cameraToCapture3.transform.eulerAngles = cameras.transform.TransformDirection(new_angle);

            Camera cam = cameraToCapture3.GetComponent<Camera>();
            float old_focalLength = cam.focalLength;
            float new_focalLength;
            /*CAMERA3_FOCALLENGTH*/ new_focalLength = old_focalLength;
            cam.focalLength = new_focalLength;
        }

        // Change parameters of CameraToCapture4
        {
            GameObject cameraToCapture4=GameObject.Find("Camera4");

            Vector3 old_pos = cameraToCapture4.transform.position;
            Vector3 new_pos;
            /*CAMERA4_POSITION_X*/ new_pos.x = old_pos.x;
            /*CAMERA4_POSITION_Y*/ new_pos.y = old_pos.y;
            /*CAMERA4_POSITION_Z*/ new_pos.z = old_pos.z;
            cameraToCapture4.transform.position = cameras.transform.TransformPoint(new_pos);

            Vector3 old_angle = cameraToCapture4.transform.eulerAngles;
            Vector3 new_angle;
            /*CAMERA4_ANGLE_X*/ new_angle.x = old_angle.x;
            /*CAMERA4_ANGLE_Y*/ new_angle.y = old_angle.y;
            /*CAMERA4_ANGLE_Z*/ new_angle.z = 0;
            cameraToCapture4.transform.eulerAngles = cameras.transform.TransformDirection(new_angle);

            Camera cam = cameraToCapture4.GetComponent<Camera>();
            float old_focalLength = cam.focalLength;
            float new_focalLength;
            /*CAMERA4_FOCALLENGTH*/ new_focalLength = old_focalLength;
            cam.focalLength = new_focalLength;
        }

        {
            GameObject cameraToCapture5=GameObject.Find("Camera5");

            Vector3 old_pos = cameraToCapture5.transform.position;
            Vector3 new_pos;
            /*CAMERA5_POSITION_X*/ new_pos.x = old_pos.x;
            /*CAMERA5_POSITION_Y*/ new_pos.y = old_pos.y;
            /*CAMERA5_POSITION_Z*/ new_pos.z = old_pos.z;
            cameraToCapture5.transform.position = cameras.transform.TransformPoint(new_pos);

            Vector3 old_angle = cameraToCapture5.transform.eulerAngles;
            Vector3 new_angle;
            /*CAMERA5_ANGLE_X*/ new_angle.x = old_angle.x;
            /*CAMERA5_ANGLE_Y*/ new_angle.y = old_angle.y;
            /*CAMERA5_ANGLE_Z*/ new_angle.z = 0;
            cameraToCapture5.transform.eulerAngles = cameras.transform.TransformDirection(new_angle);

            Camera cam = cameraToCapture5.GetComponent<Camera>();
            float old_focalLength = cam.focalLength;
            float new_focalLength;
            /*CAMERA5_FOCALLENGTH*/ new_focalLength = old_focalLength;
            cam.focalLength = new_focalLength;
        }

        {
            GameObject cameraToCapture6=GameObject.Find("Camera6");

            Vector3 old_pos = cameraToCapture6.transform.position;
            Vector3 new_pos;
            /*CAMERA6_POSITION_X*/ new_pos.x = old_pos.x;
            /*CAMERA6_POSITION_Y*/ new_pos.y = old_pos.y;
            /*CAMERA6_POSITION_Z*/ new_pos.z = old_pos.z;
            cameraToCapture6.transform.position = cameras.transform.TransformPoint(new_pos);

            Vector3 old_angle = cameraToCapture6.transform.eulerAngles;
            Vector3 new_angle;
            /*CAMERA6_ANGLE_X*/ new_angle.x = old_angle.x;
            /*CAMERA6_ANGLE_Y*/ new_angle.y = old_angle.y;
            /*CAMERA6_ANGLE_Z*/ new_angle.z = 0;
            cameraToCapture6.transform.eulerAngles = cameras.transform.TransformDirection(new_angle);

            Camera cam = cameraToCapture6.GetComponent<Camera>();
            float old_focalLength = cam.focalLength;
            float new_focalLength;
            /*CAMERA6_FOCALLENGTH*/ new_focalLength = old_focalLength;
            cam.focalLength = new_focalLength;
        }

        {
            GameObject cameraToCapture7=GameObject.Find("Camera7");

            Vector3 old_pos = cameraToCapture7.transform.position;
            Vector3 new_pos;
            /*CAMERA7_POSITION_X*/ new_pos.x = old_pos.x;
            /*CAMERA7_POSITION_Y*/ new_pos.y = old_pos.y;
            /*CAMERA7_POSITION_Z*/ new_pos.z = old_pos.z;
            cameraToCapture7.transform.position = cameras.transform.TransformPoint(new_pos);

            Vector3 old_angle = cameraToCapture7.transform.eulerAngles;
            Vector3 new_angle;
            /*CAMERA7_ANGLE_X*/ new_angle.x = old_angle.x;
            /*CAMERA7_ANGLE_Y*/ new_angle.y = old_angle.y;
            /*CAMERA7_ANGLE_Z*/ new_angle.z = 0;
            cameraToCapture7.transform.eulerAngles = cameras.transform.TransformDirection(new_angle);

            Camera cam = cameraToCapture7.GetComponent<Camera>();
            float old_focalLength = cam.focalLength;
            float new_focalLength;
            /*CAMERA7_FOCALLENGTH*/ new_focalLength = old_focalLength;
            cam.focalLength = new_focalLength;
        }

        {
            GameObject cameraToCapture8=GameObject.Find("Camera8");

            Vector3 old_pos = cameraToCapture8.transform.position;
            Vector3 new_pos;
            /*CAMERA8_POSITION_X*/ new_pos.x = old_pos.x;
            /*CAMERA8_POSITION_Y*/ new_pos.y = old_pos.y;
            /*CAMERA8_POSITION_Z*/ new_pos.z = old_pos.z;
            cameraToCapture8.transform.position = cameras.transform.TransformPoint(new_pos);

            Vector3 old_angle = cameraToCapture8.transform.eulerAngles;
            Vector3 new_angle;
            /*CAMERA8_ANGLE_X*/ new_angle.x = old_angle.x;
            /*CAMERA8_ANGLE_Y*/ new_angle.y = old_angle.y;
            /*CAMERA8_ANGLE_Z*/ new_angle.z = 0;
            cameraToCapture8.transform.eulerAngles = cameras.transform.TransformDirection(new_angle);

            Camera cam = cameraToCapture8.GetComponent<Camera>();
            float old_focalLength = cam.focalLength;
            float new_focalLength;
            /*CAMERA8_FOCALLENGTH*/ new_focalLength = old_focalLength;
            cam.focalLength = new_focalLength;
        }
    }
}
