// TO-Bour early design: running-skill agent for a legged robot on a beam.
// Reward follows the skill reward in "Robot Parkour Learning" (Zhuang, Fu et al., CoRL 2023):
// forward-speed tracking, lateral and yaw terms, an energy cost, and an alive bonus.

using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class CrawlerAgent : Agent
{
    public Transform ground;

    private Rigidbody rb;
    private EnvironmentParameters param;
    private Controller Controller;
    private Transform body;

    // Hyperparameters
    [Header("Hyperparameters")][Space(2)]
    public float alpha1;
    public float alpha2;
    public float alpha3;
    public float alpha4;
    public float alpha5;
    public float alpha6;

    [Header("Environment Variables")][Space(2)]
    public float TargetVelocity;
    public float TrackEnd;

    [Header("Reward Inspector")][Space(2)]
    public float ForwVelocityReward;
    public float LatYVelocityReward;
    public float LatZVelocityReward;
    public float YawVelocityReward;
    public float EnergyReward;
    public float YReward;
    public float ZReward;

    [Header("Transform Inspector")][Space(2)]
    public Vector3 AgentPosition;
    public Quaternion AgentRotation;

    private float[] previousAction = new float[12];
    
    void Start()
    {
        param = Academy.Instance.EnvironmentParameters;
        Controller = GetComponent<Controller>();
        
        foreach (Transform child in transform)
        {
            body = child;
            rb = body.GetComponent<Rigidbody>();
        }

        foreach (Transform child in body)
        {
            if (!child.CompareTag("leg"))
                continue;
            Controller.SetupLeg(child);
        }
    }

    public override void OnEpisodeBegin() {
        for (int i = 0; i < previousAction.Length; i++)
            previousAction[i] = 0f;
        
        ResetAgentMovement();
        ResetEnvironmentVar();
    }

    private void ResetAgentMovement() {
        foreach (BodyPart bp in Controller.bpList)
            bp.Reset();
        rb.angularVelocity = Vector3.zero;
        rb.velocity = Vector3.zero;
        body.rotation = Quaternion.Euler(0, 0, 90);
        body.localPosition = new Vector3(0, 0, 0);
    }

    private void ResetEnvironmentVar() {
        TargetVelocity = Utilities.GetTargetVelocity(param);
        TrackEnd = UnityEngine.Random.Range(35f, 45f);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        float pitch = Utilities.NormalizeAngle(body.transform.localEulerAngles.x) / 180f;
        float roll = Utilities.NormalizeAngle(body.transform.localEulerAngles.z) / 180f;
        float yaw = Utilities.NormalizeAngle(body.transform.localEulerAngles.y) / 180f;
        sensor.AddObservation(pitch);
        sensor.AddObservation(roll);
        sensor.AddObservation(yaw); // Added property, not from paper

        Vector3 localAngularVelocity = body.transform.InverseTransformDirection(rb.angularVelocity);
        sensor.AddObservation(localAngularVelocity);

        List<float> jointPositions = Controller.GetBodyPartPositions();
        List<float> jointVelocities = Controller.GetBodyPartVelocities();
        sensor.AddObservation(jointPositions);
        sensor.AddObservation(jointVelocities);

        sensor.AddObservation(previousAction);

        // E_Phys
        sensor.AddObservation(TargetVelocity);

        // Inspector Variables
        AgentPosition = body.localPosition;
        AgentRotation = body.localRotation;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Send Action
        List<BodyPart> bpList = Controller.bpList;
        ActionSegment<float> continuousActions = actions.ContinuousActions;
        for (int i = 0; i < continuousActions.Length; i++)
            bpList[i].SetJointTargetRotation(continuousActions[i]);
        previousAction = continuousActions.Array;

        // Rewards:

        // R_Forward
        float forwardVelocity = rb.velocity.x;
        float forwardVelocityReward = -alpha1 * Mathf.Abs(forwardVelocity - TargetVelocity);

        float lateralYVelocity = rb.velocity.y;
        float lateralYVelocityReward = -alpha2 * Mathf.Pow(lateralYVelocity, 2);

        float lateralZVelocity = rb.velocity.z;
        float lateralZVelocityReward = -alpha2 * Mathf.Pow(lateralZVelocity, 2);

        float yawVelocity = transform.InverseTransformDirection(rb.angularVelocity).y;
        float yawVelocityReward = alpha3 * Mathf.Exp(-Mathf.Abs(yawVelocity));

        float r_forward = forwardVelocityReward + lateralYVelocityReward + yawVelocityReward;
        AddReward(r_forward);

        // R_Energy
        float energyReward = Controller.GetBodyPartTorques().Sum();
        float r_energy = -alpha4 * energyReward;
        AddReward(r_energy);

        // Parameter Tracking
        ForwVelocityReward = forwardVelocityReward;
        LatYVelocityReward = lateralYVelocityReward;
        LatZVelocityReward = lateralZVelocityReward;
        YawVelocityReward = yawVelocityReward;
        EnergyReward = r_energy;
        YReward = 0.15f + Mathf.Min(body.localPosition.y, 0f);
        ZReward = -Math.Abs(body.localPosition.z);

        // R_Alive
        AddReward(YReward);
        AddReward(LatZVelocityReward);
        AddReward(1f);

        // Robot fell Over
        if (!Utilities.IsAngleInRange(body.transform.eulerAngles.x, -65, 65)
         || !Utilities.IsAngleInRange(body.transform.eulerAngles.z, 60, 110)
         || !Utilities.IsAngleInRange(body.transform.eulerAngles.y, -45, 45))
        {
            SetReward(-10f);
            EndEpisode();
        }

        // Robot Fell Off Platform
        if (body.transform.position.y < 0.5f)
        {
            SetReward(-10f);
            EndEpisode();
        }

        // Robot reached platform edge
        if (Mathf.Abs(body.transform.localPosition.z) > 1f)
        {
            SetReward(-10f);
            EndEpisode();
        }

        // Robot Reached End of Track
        if (body.transform.localPosition.x > TrackEnd)
        {
            TrackEnd = 9999;
            TargetVelocity = 0;
            AddReward(10f);
        }

        // Robot Reached End of Platform
        if (body.transform.localPosition.x > 51f)
        {
            SetReward(-10f);
            EndEpisode();
        }
    }

    // Manual testing: numpad keys 1-9 pick a joint group (rows of three keys)
    // and a target within that group (neutral, max, min).
    private static readonly KeyCode[] HeuristicKeys =
    {
        KeyCode.Keypad1, KeyCode.Keypad2, KeyCode.Keypad3,
        KeyCode.Keypad4, KeyCode.Keypad5, KeyCode.Keypad6,
        KeyCode.Keypad7, KeyCode.Keypad8, KeyCode.Keypad9,
    };
    private static readonly float[] HeuristicTargets = { 0f, 1f, -1f };

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        List<BodyPart> bpList = Controller.bpList;

        for (int k = 0; k < HeuristicKeys.Length; k++)
        {
            if (!Input.GetKey(HeuristicKeys[k]))
                continue;

            int jointGroup = k / 3;
            float target = HeuristicTargets[k % 3];
            for (int i = 0; i < bpList.Count; i++)
            {
                if (i % 3 == jointGroup)
                    bpList[i].SetJointTargetRotation(target);
            }
            return;
        }
    }
}