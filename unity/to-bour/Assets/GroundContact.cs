// Adapted from Unity ML-Agents (Examples/SharedAssets/Scripts/GroundContact.cs),
// Copyright Unity Technologies, licensed under the Apache License 2.0 (see THIRD_PARTY_NOTICES.md).
// Changes: removed the namespace and the reset-on-contact option.

using UnityEngine;
using Unity.MLAgents;

public class GroundContact : MonoBehaviour
{
    public Agent agent;

    [Header("Ground Check")]
    public bool penalizeGroundContact; // Whether to penalize on contact.
    public float groundContactPenalty; // Penalty amount (ex: -1).
    public bool touchingGround;
    const string k_Ground = "ground"; // Tag of ground object.

    /// <summary>
    /// Check for collision with ground, and optionally penalize agent.
    /// </summary>
    void OnCollisionEnter(Collision col)
    {
        if (col.transform.CompareTag(k_Ground))
        {
            touchingGround = true;
            if (penalizeGroundContact)
            {
                agent.AddReward(groundContactPenalty);
            }
        }
    }

    /// <summary>
    /// Check for end of ground collision and reset flag appropriately.
    /// </summary>
    void OnCollisionExit(Collision other)
    {
        if (other.transform.CompareTag(k_Ground))
        {
            touchingGround = false;
        }
    }
}
