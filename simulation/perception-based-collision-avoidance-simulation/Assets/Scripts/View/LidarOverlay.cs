using UnityEngine;

// The last lidar scan as world-space points, and the GL drawing of dots, beams, ground-truth boxes and the route in the player's camera.
// Debug view only: it reads what the lidar sensor already computed, and it draws only into cameras that render to the screen (never into the
// sensor camera, which renders to a texture), so nothing here can reach the processor or its camera images.
public class LidarOverlay
{
    // the last scan, valid returns only
    public Vector3[] Points = new Vector3[0];
    public byte[] Class = new byte[0];      // 1 ground-like surface, 2 other surface
    public int[] Cell = new int[0];         // index in the range image: row * Cols + column
    public int Count;
    public Vector3 Origin;
    public int Rows, Cols, FrameId = -1;

    static readonly Color GroundColor = new Color(0.25f, 0.85f, 0.35f, 0.9f);
    static readonly Color ObjectColor = new Color(1f, 0.28f, 0.12f, 1f);
    static readonly Color RingColor = new Color(0.2f, 0.9f, 1f, 0.85f);
    static readonly Color BeamColor = new Color(1f, 0.95f, 0.4f, 0.35f);
    static readonly Color BoxColor = new Color(0.2f, 0.9f, 1f, 1f);
    static readonly Color RouteColor = new Color(1f, 0.85f, 0.1f, 1f);

    // Rebuilds the point list when a new scan has arrived. Cheap: at most 32 x 600 cells, ten times a second.
    public void Refresh(LidarSensor lidar)
    {
        LidarScan scan = lidar.LastScan;
        if (scan == null || scan.frameId == FrameId) return;
        FrameId = scan.frameId;
        Rows = scan.rows;
        Cols = scan.cols;
        int n = scan.ranges.Length;
        if (Points.Length < n)
        {
            Points = new Vector3[n];
            Class = new byte[n];
            Cell = new int[n];
        }
        Origin = lidar.LastOrigin;
        int k = 0;
        for (int i = 0; i < n; i++)
        {
            ushort v = scan.ranges[i];
            if (v == LidarScan.NoReturn || v == LidarScan.MaxRangeOrSky) continue;
            Points[k] = Origin + lidar.CellDirectionWorld(i) * (v * 0.01f);
            Class[k] = lidar.CellClass(i);
            Cell[k] = i;
            k++;
        }
        Count = k;
    }

    // Small squares that face the camera, on the surfaces the lidar hit. Objects red, ground green.
    public void DrawDots(Camera cam, ViewControls.DotMode mode)
    {
        Vector3 camPos = cam.transform.position;
        Vector3 right = cam.transform.right, up = cam.transform.up;
        GL.Begin(GL.QUADS);
        for (int k = 0; k < Count; k++)
        {
            byte c = Class[k];
            if (mode == ViewControls.DotMode.Objects && c != 2) continue;
            Vector3 p = Points[k];
            Vector3 toCam = camPos - p;
            float dist = toCam.magnitude;
            if (dist < 0.5f) continue;
            p += toCam * (0.06f / dist);                         // a little toward the camera so the square is not cut by the surface it sits on
            float h = Mathf.Clamp(dist * 0.004f, 0.05f, 0.5f);   // grows with distance so far dots stay visible
            Vector3 r = right * h, u = up * h;
            GL.Color(c == 1 ? GroundColor : ObjectColor);
            GL.Vertex(p - r - u);
            GL.Vertex(p - r + u);
            GL.Vertex(p + r + u);
            GL.Vertex(p + r - u);
        }
        GL.End();
    }

    // Lines from the lidar to the hits. All 19 200 at once would be a solid mess, so every mode draws a small selection:
    //   Ring     the returns of one beam row (one sweep around the car)
    //   Sparse   every sparseColStep-th column of every sparseRowStep-th row
    //   Objects  returns on non-ground surfaces, every stride-th one
    public int DrawBeams(ViewControls.BeamMode mode, int ringRow, int sparseColStep, int sparseRowStep, int objectStride, int maxBeams)
    {
        if (Cols <= 0) return 0;
        sparseColStep = Mathf.Max(1, sparseColStep);
        sparseRowStep = Mathf.Max(1, sparseRowStep);
        objectStride = Mathf.Max(1, objectStride);
        int drawn = 0, objectsSeen = 0;
        GL.Begin(GL.LINES);
        for (int k = 0; k < Count && drawn < maxBeams; k++)
        {
            int cell = Cell[k];
            int row = cell / Cols, col = cell % Cols;
            Color color = BeamColor;
            if (mode == ViewControls.BeamMode.Ring)
            {
                if (row != ringRow) continue;
                color = RingColor;
            }
            else if (mode == ViewControls.BeamMode.Sparse)
            {
                if (row % sparseRowStep != 0 || col % sparseColStep != 0) continue;
            }
            else
            {
                if (Class[k] != 2) continue;
                if (objectsSeen++ % objectStride != 0) continue;
            }
            GL.Color(color);
            GL.Vertex(Origin);
            GL.Vertex(Points[k]);
            drawn++;
        }
        GL.End();
        return drawn;
    }

    // Wire boxes around the Hazard objects: the ground truth the system is supposed to find. The box follows the hazard's own transform.
    public static void DrawHazardBoxes()
    {
        GL.Begin(GL.LINES);
        GL.Color(BoxColor);
        foreach (Hazard h in Hazard.All)
        {
            if (h == null || h.Length <= 0f) continue;
            Quaternion q = h.transform.rotation;
            Vector3 c = h.WorldCenter;
            Vector3 half = new Vector3(h.Width, h.Height, h.Length) * 0.5f;
            Vector3[] v = new Vector3[8];
            for (int i = 0; i < 8; i++)
            {
                Vector3 s = new Vector3((i & 1) == 0 ? -1f : 1f, (i & 2) == 0 ? -1f : 1f, (i & 4) == 0 ? -1f : 1f);
                v[i] = c + q * Vector3.Scale(half, s);
            }
            for (int i = 0; i < 8; i++)
            {
                if ((i & 1) == 0) { GL.Vertex(v[i]); GL.Vertex(v[i | 1]); }
                if ((i & 2) == 0) { GL.Vertex(v[i]); GL.Vertex(v[i | 2]); }
                if ((i & 4) == 0) { GL.Vertex(v[i]); GL.Vertex(v[i | 4]); }
            }
        }
        GL.End();
    }

    public static void DrawPolyline(Vector3[] pts)
    {
        if (pts == null || pts.Length < 2) return;
        GL.Begin(GL.LINES);
        GL.Color(RouteColor);
        for (int i = 0; i + 1 < pts.Length; i++)
        {
            GL.Vertex(pts[i]);
            GL.Vertex(pts[i + 1]);
        }
        GL.End();
    }
}