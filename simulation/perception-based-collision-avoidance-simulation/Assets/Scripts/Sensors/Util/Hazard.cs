using System.Collections.Generic;
using UnityEngine;

// Marks a scenario object that can be hit or that matters for the evaluation.
// GROUND TRUTH ONLY: nothing in this component (label, size, velocity) ever goes into a message to the processor.
[DefaultExecutionOrder(10)] // after the scripts that move objects (order 0), before the loggers
public class Hazard : MonoBehaviour
{
    public static readonly List<Hazard> All = new List<Hazard>();
    static int nextId = 1;

    public string label = "obstacle";           // ground-truth log only

    public int Id { get; private set; }
    public float Length { get; private set; }   // along the object's forward axis (local z)
    public float Width { get; private set; }    // local x
    public float Height { get; private set; }   // local y
    public Vector3 WorldVelocity { get; private set; }     // m/s, finite difference per physics step
    public readonly List<int> ColliderIds = new List<int>();

    Collider primary;
    Vector3 lastCentre;
    bool hasLast;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetStatics() { All.Clear(); nextId = 1; }

    void Awake()
    {
        Id = nextId++;
        foreach (Collider c in GetComponentsInChildren<Collider>())
        {
            ColliderIds.Add(c.GetInstanceID());
            if (primary == null) primary = c;
        }
        if (primary == null) { UnityEngine.Debug.LogWarning("Hazard '" + name + "' has no collider"); return; }

        Vector3 sc = primary.transform.lossyScale;
        var box = primary as BoxCollider;
        var cap = primary as CapsuleCollider;   // assumed to stand upright
        if (box != null) { Length = box.size.z * sc.z; Width = box.size.x * sc.x; Height = box.size.y * sc.y; }
        else if (cap != null) { Length = Width = 2f * cap.radius * Mathf.Max(sc.x, sc.z); Height = cap.height * sc.y; }
        else { Vector3 b = primary.bounds.size; Length = b.z; Width = b.x; Height = b.y; }
    }

    void OnEnable() { All.Add(this); }
    void OnDisable() { All.Remove(this); }

    void Start()
    {
        int layer = LayerMask.NameToLayer("Hazard");
        if (layer >= 0 && gameObject.layer != layer)
            UnityEngine.Debug.LogWarning("Hazard '" + name + "' is not on the Hazard layer, so the ego car will not collide with it");
    }

    public Vector3 WorldCenter
    {
        get
        {
            var box = primary as BoxCollider;
            if (box != null) return primary.transform.TransformPoint(box.center);
            var cap = primary as CapsuleCollider;
            if (cap != null) return primary.transform.TransformPoint(cap.center);
            return primary != null ? primary.bounds.center : transform.position;
        }
    }

    void FixedUpdate()
    {
        Vector3 c = WorldCenter;
        WorldVelocity = hasLast ? (c - lastCentre) / Time.fixedDeltaTime : Vector3.zero;
        lastCentre = c;
        hasLast = true;
    }
}