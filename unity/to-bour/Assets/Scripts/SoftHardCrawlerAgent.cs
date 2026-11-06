// TO-Bour agent: the Blender-modeled robot with obstacle courses and a soft/hard dynamics toggle,
// following the two-stage setup in "Robot Parkour Learning" (Zhuang, Fu et al., CoRL 2023).
// With soft dynamics on, obstacles are drawn black and the robot can pass through them.

using System.Collections.Generic;
using System.Linq;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class SoftHardCrawlerAgent : Agent
{
    public Controller Controller;
    public Rigidbody rb;

    [Header("Environment Settings")]
    [Space(2)]
    public GameObject blockPrefab;
    public Material blackColor;
    public Material whiteColor;
    public string obstacleType;
    public bool useSoftDynamics;

    public float ObstacleLength;
    public float ObstacleWidth;

    [Header("Body Parts")][Space(10)]
    public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;


    // Hyperparameters
    [Header("Hyperparameters")]
    [Space(2)]
    public float alpha1;
    public float alpha2;
    public float alpha3;
    public float alpha4;
    public float alpha5;
    public float alpha6;

    [Header("Variables")]
    [Space(2)]
    public float TargetVelocity;

    [Header("Reward Inspector")]
    [Space(2)]
    public float ForwVelocityReward;
    public float LatYVelocityReward;
    public float LatZVelocityReward;
    public float YawVelocityReward;
    public float EnergyReward;
    public float YReward;
    public float ZReward;

    [Header("Transform Inspector")]
    [Space(2)]
    public Vector3 AgentPosition;
    public Quaternion AgentRotation;

    private float[] previousAction = new float[8];

    private GameObject[] environmentGameObjects = new GameObject[4];

    public override void Initialize()
    {
        Controller.SetupBodyPart(leg0Upper);
        Controller.SetupBodyPart(leg0Lower);
        Controller.SetupBodyPart(leg1Upper);
        Controller.SetupBodyPart(leg1Lower);
        Controller.SetupBodyPart(leg2Upper);
        Controller.SetupBodyPart(leg2Lower);
        Controller.SetupBodyPart(leg3Upper);
        Controller.SetupBodyPart(leg3Lower);
    }

    public override void OnEpisodeBegin()
    {
        for (int i = 0; i < previousAction.Length; i++)
        {
            previousAction[i] = 0f;
        }

        foreach (var bodyPart in Controller.bpList)
        {
            bodyPart.Reset();
        }
        ResetEnvironment();
    }
    // Privileged obstacle information: obstacle center, block length, block height
    public float obstaclePositionX;
    public float obstacleBlockLength;
    public float obstacleBlockHeight;

    private void ResetEnvironment()
    {
        for (int i = 0; i < environmentGameObjects.Length; i++)
        {
            if (environmentGameObjects[i] != null)
            {
                Destroy(environmentGameObjects[i]);
            }
            environmentGameObjects[i] = null;
        }
        Vector3 obstacleOrigin = new Vector3(transform.localPosition.x + ObstacleLength / 4, transform.localPosition.y, transform.localPosition.z);
        obstaclePositionX = obstacleOrigin.x;
        switch (obstacleType)
        {
            case "RUNNING":
                {
                    obstacleBlockLength = ObstacleLength;
                    obstacleBlockHeight = 0;
                    break;
                }
            case "LEAPING":
                {
                    SpawnLeap(obstacleOrigin, maxLeapGap);
                    obstacleBlockLength = maxLeapGap;
                    obstacleBlockHeight = 0;
                    break;
                }
            case "CLIMBING":
                {
                    SpawnClimb(obstacleOrigin, maxClimbHeight);
                    obstacleBlockLength = blockLength;
                    obstacleBlockHeight = maxClimbHeight;
                    break;
                }
            case "CRAWLING":
                {
                    SpawnCrawl(obstacleOrigin, maxCrawlHeight);
                    obstacleBlockLength = blockLength;
                    obstacleBlockHeight = maxCrawlHeight;
                    break;
                }
            default:
                {
                    Debug.LogWarning("Invalid obstacle type: " + obstacleType);
                    break;
                }
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // pitch, roll, yaw of body (3)
        float pitch = Utilities.NormalizeAngle(body.transform.localEulerAngles.x) / 180f;
        float roll = Utilities.NormalizeAngle(body.transform.localEulerAngles.z) / 180f;
        float yaw = Utilities.NormalizeAngle(body.transform.localEulerAngles.y) / 180f;
        sensor.AddObservation(pitch);
        sensor.AddObservation(roll);
        sensor.AddObservation(yaw); // Added property, not from paper

        // joint positions and velocities (8 + 8 = 16)
        List<float> jointPositions = Controller.GetBodyPartPositions();
        List<float> jointVelocities = Controller.GetBodyPartVelocities();
        sensor.AddObservation(jointPositions);
        sensor.AddObservation(jointVelocities);

        // previous actions (8)
        sensor.AddObservation(previousAction);

        // target velocity (1)
        sensor.AddObservation(TargetVelocity);

        // privileged visual info (3)
        // distance from middle of obstacle, length of block obstacle, height of block obstacle
        sensor.AddObservation(obstaclePositionX -  body.transform.localPosition.x);
        sensor.AddObservation(obstacleBlockLength);
        sensor.AddObservation(obstacleBlockHeight);

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
        {
            bpList[i].SetJointTargetRotation(continuousActions[i]);
        }
        previousAction = continuousActions.Array;

        // Rewards:

        // R_Forward
        float forwardVelocity = rb.velocity.x;
        float forwardVelocityReward = -alpha1 * Mathf.Abs(forwardVelocity - TargetVelocity);

        float lateralYVelocity = rb.velocity.y;
        float lateralYVelocityReward = -alpha2 * Mathf.Pow(lateralYVelocity, 2);

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
        YawVelocityReward = yawVelocityReward;
        EnergyReward = r_energy;

        // R_Alive: end the episode if the robot falls or moves implausibly fast
        if (body.transform.position.y < 0 || rb.velocity.magnitude > 5)
        {
            AddReward(-500);
            EndEpisode();
        }
        else
        {
            AddReward(0.01f);
        }

        // End the episode at the end of the track
        if (body.transform.position.x > transform.position.x + ObstacleLength / 2)
        {
            EndEpisode();
        }
    }

    // Manual testing: number keys set every joint to the same target.
    // Keys 1-5 give -0.2 to -0.99; keys 6-9 and 0 give 0.99 down to 0.2.
    private static readonly KeyCode[] HeuristicKeys =
    {
        KeyCode.Alpha1, KeyCode.Alpha2, KeyCode.Alpha3, KeyCode.Alpha4, KeyCode.Alpha5,
        KeyCode.Alpha6, KeyCode.Alpha7, KeyCode.Alpha8, KeyCode.Alpha9, KeyCode.Alpha0,
    };
    private static readonly float[] HeuristicTargets =
    {
        -0.2f, -0.4f, -0.6f, -0.8f, -0.99f,
        0.99f, 0.8f, 0.6f, 0.4f, 0.2f,
    };

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;

        float target = 0f;
        for (int k = 0; k < HeuristicKeys.Length; k++)
        {
            if (Input.GetKey(HeuristicKeys[k]))
            {
                target = HeuristicTargets[k];
                break;
            }
        }

        for (int i = 0; i < 8; i++)
        {
            continuousActionsOut[i] = target;
        }
    }

    /*
        ENVIRONMENT GENERATION (obstacle courses by Sarah Ibadi)
        Leap, climb, and crawl obstacles. With soft dynamics on, the obstacle blocks are
        penetrable and drawn black; with hard dynamics they are solid and white.
    */
    private float minLeapGap = 0.05f;
    private float maxLeapGap = 0.1f;
    private float minCrawlHeight = 0.16f; // min as in the easiest
    private float maxCrawlHeight = 0.13f;
    private float minClimbHeight = 0.01f;
    private float maxClimbHeight = 0.1f;

    private float blockLength = 0.15f;

    private float penetrableLeapHeight = 0.02f;

    GameObject SpawnBlock(Vector3 localPosition, Vector3 size, bool isPenetrable)
    {
        GameObject block = Instantiate(blockPrefab, Vector3.zero, Quaternion.identity);

        block.transform.localScale = size;
        block.transform.SetParent(transform);
        block.transform.localPosition = localPosition;
        block.transform.localRotation = Quaternion.identity;

        block.GetComponent<Renderer>().material = isPenetrable ? blackColor : whiteColor;
        block.GetComponent<Collider>().enabled = !isPenetrable;
        return block;
    }

    void SpawnLeap(Vector3 localPosition, float gapLength)
    {
        GameObject block1 = SpawnBlock(new Vector3(localPosition.x, localPosition.y - 1.5f, localPosition.z),
                   new Vector3(ObstacleLength, 1.0f, ObstacleWidth), false);

        float d = ((ObstacleLength - gapLength) / 4 + gapLength / 2);

        GameObject block2 = SpawnBlock(new Vector3(localPosition.x - d, localPosition.y - 0.5f, localPosition.z),
                   new Vector3((ObstacleLength - gapLength) / 2, 1.0f, ObstacleWidth), false);
        GameObject block3 = SpawnBlock(new Vector3(localPosition.x + d, localPosition.y - 0.5f, localPosition.z),
                   new Vector3((ObstacleLength - gapLength) / 2, 1.0f, ObstacleWidth), false);

        if(useSoftDynamics)
        {
            GameObject block4 = SpawnBlock(new Vector3(localPosition.x, localPosition.y + penetrableLeapHeight/2, localPosition.z),
                   new Vector3(gapLength, 0.1f, ObstacleWidth), true);
            environmentGameObjects[3] = block4;
        }

        environmentGameObjects[0] = block1;
        environmentGameObjects[1] = block2;
        environmentGameObjects[2] = block3;
    }

    void SpawnCrawl(Vector3 localPosition, float height)
    {
        GameObject block1 = SpawnBlock(new Vector3(localPosition.x, localPosition.y - 0.5f, localPosition.z),
                   new Vector3(ObstacleLength, 1.0f, ObstacleWidth), false);

        GameObject block2 = SpawnBlock(new Vector3(localPosition.x, localPosition.y + height + 2.5f, localPosition.z),
                   new Vector3(blockLength, 5.0f, ObstacleWidth), useSoftDynamics);
        environmentGameObjects[0] = block1;
        environmentGameObjects[1] = block2;
    }

    void SpawnClimb(Vector3 localPosition, float height)
    {
        GameObject block1 = SpawnBlock(new Vector3(localPosition.x, localPosition.y - 0.5f, localPosition.z),
                   new Vector3(ObstacleLength, 1.0f, ObstacleWidth), false);

        GameObject block2 = SpawnBlock(new Vector3(localPosition.x, localPosition.y + height / 2, localPosition.z),
                   new Vector3(blockLength, height, ObstacleWidth), useSoftDynamics);
        environmentGameObjects[0] = block1;
        environmentGameObjects[1] = block2;
    }

    void SpawnRun(Vector3 localPosition)
    {
        GameObject block1 = SpawnBlock(new Vector3(localPosition.x, localPosition.y - 0.5f, localPosition.z),
                   new Vector3(ObstacleLength, 1.0f, ObstacleWidth), false);
        block1.gameObject.tag = "ground";
        environmentGameObjects[0] = block1;
    }

    float CalculateGapFromScore(float score)
    {
        return Mathf.Lerp(minLeapGap, maxLeapGap, score);
    }

    float CalculateCrawlHeightFromScore(float score)
    {
        return Mathf.Lerp(maxCrawlHeight, minCrawlHeight, score);
    }

    float CalculateClimbHeightFromScore(float score)
    {
        return Mathf.Lerp(minClimbHeight, maxClimbHeight, score);
    }
}