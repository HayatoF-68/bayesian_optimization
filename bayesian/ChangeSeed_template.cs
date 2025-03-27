using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AWSIM.TrafficSimulation;

[DefaultExecutionOrder(-1)]
public class ChangeSeed: MonoBehaviour {

    public TrafficManager trafficManager;
    int old_seed;

    void Start()
    {
        /*SEED*/trafficManager.seed = old_seed;
    }
}
