// Copies tests/fixtures/api into public/fixtures so the UI can run without a backend.
import { cpSync, mkdirSync } from "node:fs";

mkdirSync("public/fixtures", { recursive: true });
cpSync("../tests/fixtures/api", "public/fixtures", { recursive: true });
