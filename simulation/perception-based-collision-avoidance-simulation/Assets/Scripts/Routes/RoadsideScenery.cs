using System.Collections.Generic;
using UnityEngine;

// Settings for the scenery along a road: buildings on both sides, trees in the verge behind the sidewalks, street lamps at the curb and small
// props on the sidewalks. Put it on the road object (next to RouteRoadMesh) and run Tools > Environment Kit > Generate roadside scenery.
// The scenery is made IN THE EDITOR as ordinary objects under a child called "Generated": they are saved in the scene, you can move or delete single
// ones by hand, and generating again with the same settings gives the same result (seed) and replaces everything under "Generated".
// This component does nothing at run time.
public class RoadsideScenery : MonoBehaviour
{
    [Header("Road (copied from RouteRoadMesh when one is on this object)")]
    public RouteRoadMesh roadMesh;
    public string routeFile = "routes/dev_road.json";
    public float roadWidth = 7f;
    public float sidewalkWidth = 3f;        // copied from RouteRoadDressing when there is one on this object
    public float centreOffsetM = 0f;        // copied from RouteRoadMesh: the road is this far to the left of the route

    [Header("Range")]
    public int seed = 1;
    public float startM = 0f;
    public float endM = 0f;                  // 0 = to the end of the route
    public List<Vector2> keepClearM = new List<Vector2>();     // (from, to) in metres along the route: nothing is placed there, on either side

    [Header("Buildings (front toward the road)")]
    public bool buildingsLeft = true;
    public bool buildingsRight = true;
    public string buildingFilter = "";       // prefab name prefix: "PD_" Pandazole only, "SP_" SimplePoly only, empty = both
    public List<string> buildingExclude = new List<string> {      // prefabs that are never used (the biggest SimplePoly buildings)
        "SP_building_28", "SP_building_29", "SP_building_30", "SP_building_31",
        "SP_building_32", "SP_building_33", "SP_building_34", "SP_building_35" };
    public float buildingSetbackM = 0f;      // from the outer edge of the sidewalk (0 = the front touches the sidewalk)
    public float buildingGapMin = 0.5f;
    public float buildingGapMax = 3f;
    public float buildingFrontYaw = 0f;      // turn if the models' fronts do not face the road (try 90, 180, 270)

    [Header("Trees (on the sidewalk, near the curb)")]
    public bool treesLeft = true;
    public bool treesRight = true;
    public string treeFilter = "";
    public float treeSpacingM = 14f;
    public float treeJitterM = 4f;
    public float treeCurbOffsetM = 1.6f;     // distance of the trunk from the curb

    [Header("Street lamps (alternate sides)")]
    public string lampPrefab = "SP_street_light_01";
    public float lampSpacingM = 30f;
    public float lampOffsetM = 0.7f;         // from the curb, on the sidewalk
    public float lampFrontYaw = 0f;          // turn if the lamp arm does not point over the road

    [Header("Sidewalk props (bins, hydrants, benches, mail boxes)")]
    public float propsPer100M = 3f;
}