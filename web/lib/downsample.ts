/**
 * Bucketed min/max downsampling. Keeps the first and last point, the min and max of every bucket, and every
 * index in `keep` (fault points, the focused point). Order is preserved.
 */
export function downsample<T>(items: T[], value: (t: T) => number | null | undefined, target: number, keep: (t: T) => boolean): T[] {
  if (items.length <= target) return items;
  const buckets = Math.max(1, Math.floor(target / 2));
  const size = items.length / buckets;
  const pick = new Set<number>([0, items.length - 1]);
  for (let b = 0; b < buckets; b++) {
    const lo = Math.floor(b * size);
    const hi = Math.min(items.length, Math.floor((b + 1) * size));
    let min = -1;
    let max = -1;
    for (let i = lo; i < hi; i++) {
      const v = value(items[i]);
      if (v === null || v === undefined) continue;
      if (min < 0 || v < (value(items[min]) as number)) min = i;
      if (max < 0 || v > (value(items[max]) as number)) max = i;
    }
    if (min >= 0) pick.add(min);
    if (max >= 0) pick.add(max);
  }
  items.forEach((t, i) => {
    if (keep(t)) pick.add(i);
  });
  return [...pick].sort((a, b) => a - b).map((i) => items[i]);
}
