// Joint-drive controller: configures each limb's ConfigurableJoint drive and exposes
// joint positions, velocities, and torques for observations and the energy cost.

using System.Collections.Generic;
using UnityEngine;

public class Controller : MonoBehaviour
{
    public float positionSpring; // 100
    public float positionDamper; // 10
    public float maximumForce; // 40000

    public List<BodyPart> bpList = new();

    public void SetupLeg(Transform leg)
    {
        foreach (Transform limb in leg)
            SetupBodyPart(limb);
    }

    public void SetupBodyPart(Transform limb)
    {
        BodyPart bp = new()
        {
            transform = limb,
            rb = limb.GetComponent<Rigidbody>(),
            joint = limb.GetComponent<ConfigurableJoint>(),
            startingPos = limb.localPosition,
            startingRot = limb.localRotation,
        };

        var jd = new JointDrive
        {
            positionSpring = positionSpring,
            positionDamper = positionDamper,
            maximumForce = maximumForce,
        };
        bp.joint.angularXDrive = jd;

        bpList.Add(bp);
    }

    public List<float> GetBodyPartVelocities()
    {
        List<float> velocities = new();
        foreach (BodyPart bp in bpList)
            velocities.Add(bp.GetLocalVelocity());
        return velocities;
    }

    public List<float> GetBodyPartPositions()
    {
        List<float> positions = new();
        foreach (BodyPart bp in bpList)
            positions.Add(bp.GetLocalPosition());
        return positions;
    }

    public List<float> GetBodyPartTorques()
    {
        List<float> torques = new();
        foreach (BodyPart bp in bpList)
            torques.Add(bp.joint.currentTorque.magnitude);
        return torques;
    }
}

public class BodyPart
{
    public Transform transform; 
    public ConfigurableJoint joint;
    public Rigidbody rb;
    public Vector3 startingPos;
    public Quaternion startingRot;

    public void SetJointTargetRotation(float x)
    {
        // Normalizes values between [-1, 1] to [0, 1]
        x = (x + 1f) * 0.5f;
        float xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, x);

        // Apply rotation
        joint.targetRotation = Quaternion.Euler(xRot, 0, 0);
    }

    public float GetLocalPosition() {
        // Get current x rotation
        float xRot = Utilities.NormalizeAngle(joint.targetRotation.eulerAngles.x);

        // Normalize x rotation to [-1, 1] based on angular limits
        float minLimit = joint.lowAngularXLimit.limit;
        float maxLimit = joint.highAngularXLimit.limit;

        float x = (xRot - minLimit) / (maxLimit - minLimit) * 2f - 1f;
        return x;
    }

    public float GetLocalVelocity() {
        Vector3 localAngularVelocity = transform.InverseTransformDirection(rb.angularVelocity);
        Vector3 jointAxis = joint.axis;
        
        float angularVelocity = Vector3.Dot(localAngularVelocity, jointAxis);
        return angularVelocity;
    }

    public void Reset() {
        transform.localPosition = startingPos;
        transform.localRotation = startingRot;
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        SetJointTargetRotation(0f);
    }
}