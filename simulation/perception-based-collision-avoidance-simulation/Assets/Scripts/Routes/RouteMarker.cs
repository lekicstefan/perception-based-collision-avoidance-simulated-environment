using UnityEngine;

public class RouteMarker : MonoBehaviour
{
    public string routeFile = "routes/test_route.json";
    public float speed = 10f;

    Route route;
    float s;

    void Start()
    {
        route = RouteIO.Build(RouteIO.Read(ConfigPaths.Resolve(routeFile)));
    }

    void FixedUpdate()
    {
        s = Mathf.Min(s + speed * Time.fixedDeltaTime, route.Length);
        RouteSample smp = route.Sample(s);
        transform.SetPositionAndRotation(smp.position,
            Quaternion.Euler(-Mathf.Atan(smp.grade) * Mathf.Rad2Deg, smp.YawDeg, 0f));
    }
}