// Copies tests/fixtures/api into public/fixtures so the UI can run without a backend.
// Mapping (see lib/api.ts fixtureFor): <endpoint> -> /fixtures/<file>.json, e.g.
//   /trays -> trays.json, /trays/B -> tray_B.json, /trays/B/frames -> frames_B.json,
//   /trays/B/series?metric=m -> series_B_m.json, /assessments/x -> assessment.json,
//   /assessments/x/stream -> assessment_stream.json, /evidence/x -> evidence.json[x], /eval/latest -> eval_latest.json
import { cpSync, mkdirSync } from "node:fs";

mkdirSync("public/fixtures", { recursive: true });
cpSync("../tests/fixtures/api", "public/fixtures", { recursive: true });

// Manifest so the fixture client never requests a file that does not exist (avoids 404 console noise).
import { readdirSync, writeFileSync } from "node:fs";
writeFileSync("public/fixtures/_index.json", JSON.stringify(readdirSync("public/fixtures").filter((f) => f !== "_index.json")));
