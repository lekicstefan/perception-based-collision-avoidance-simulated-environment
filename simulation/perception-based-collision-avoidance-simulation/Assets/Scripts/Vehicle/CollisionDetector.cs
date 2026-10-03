using UnityEngine;

// Detects the ego car overlapping a hazard by testing the oriented box every physics step.
public class CollisionDetector : MonoBehaviour
{
    public Vehicle vehicle;
    public LayerMask hazardMask;
    public BoxCollider box;   // found automatically on this object if left empty

    void Start()
    {
        if (box == null) box = GetComponent<BoxCollider>();
        RunControl.Reset();
    }

    void FixedUpdate()
    {
        if (RunControl.Ended) return;

        Physics.SyncTransforms(); // hazards moved by scripts in this step must be current
        Transform t = box.transform;
        Vector3 center = t.TransformPoint(box.center);
        Vector3 half = Vector3.Scale(box.size, t.lossyScale) * 0.5f;

        if (!Physics.CheckBox(center, half, t.rotation, hazardMask, QueryTriggerInteraction.Ignore)) return;

        Collider[] hits = Physics.OverlapBox(center, half, t.rotation, hazardMask, QueryTriggerInteraction.Ignore);
        string hazardName = hits.Length > 0 ? hits[0].name : "unknown";
        float impact = vehicle.Speed;
        RunControl.EndWithCollision(Time.fixedTimeAsDouble, impact, hazardName, vehicle.transform.position);
        vehicle.Freeze();
        UnityEngine.Debug.Log("COLLISION at t=" + Time.fixedTimeAsDouble.ToString("F2") + " s | impact speed " +
            impact.ToString("F2") + " m/s (" + (impact * 3.6f).ToString("F1") + " km/h) | hazard '" + hazardName +
            "' | " + hits.Length + " collider(s) overlapped");
    }

    void OnGUI()
    {
        if (!RunControl.Ended || RunControl.Reason != RunControl.EndReason.Collision) return;
        GUI.Label(new Rect(Screen.width * 0.5f - 160f, 10f, 360f, 25f),
            "COLLISION: impact speed " + RunControl.ImpactSpeed.ToString("F2") + " m/s (" +
            (RunControl.ImpactSpeed * 3.6f).ToString("F1") + " km/h) with " + RunControl.HazardName);
    }
}