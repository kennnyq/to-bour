using UnityEngine;
using Unity.MLAgents;

public static class Utilities
{
    /// <summary>
    /// Normalizes an angle to be between -180 and 180 degrees.
    /// </summary>
    public static float NormalizeAngle(float angle)
    {
        return Mathf.Repeat(angle + 180f, 360f) - 180f;
    }

    public static bool IsAngleInRange(float angle, float min, float max)
    {
        angle = NormalizeAngle(angle);
        return angle >= min && angle <= max;
    }

    public static float GetTargetVelocity(EnvironmentParameters param)
    {
        float targetVelocity = param.GetWithDefault("target_velocity", 2.0f);
        if (targetVelocity < 0.5f)
            return 0;
        return targetVelocity;
    }

    public static Vector3 GetGroundWidth(EnvironmentParameters param)
    {
        float width = param.GetWithDefault("ground_width", 1.0f);
        float num = 18.1f * (2 * width + 0.1f);
        float den = 2.1f * (20.1f - 2 * width);
        float res = 1 + 1 / Mathf.PI * Mathf.Log(num / den);
        return new Vector3(20f, 0.5f, res);
    }
}