//! Caption/subtitle track discovery from container metadata (no decoding).
//!
//! MP4/MOV: walks `moov/trak/mdia` for subtitle handlers (`sbtl`, `subt`,
//! `text`, `clcp`) or text sample entries (`tx3g`, `wvtt`, `stpp`, `c608`, ...).
//! Matroska/WebM: walks `Segment/Tracks/TrackEntry` for TrackType 17.
//!
//! Not covered: CEA-608/708 captions embedded in H.264 SEI messages, and
//! sidecar files (.srt/.vtt) that never enter the container.

use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TextTrack {
    pub codec: String,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ContainerProbe {
    pub format: &'static str,
    pub text_tracks: Vec<TextTrack>,
}

pub fn probe(bytes: &[u8]) -> ContainerProbe {
    if bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        ContainerProbe { format: "matroska", text_tracks: matroska::text_tracks(bytes) }
    } else if bytes.len() >= 8 && matches!(&bytes[4..8], b"ftyp" | b"moov" | b"mdat" | b"free" | b"wide" | b"skip") {
        ContainerProbe { format: "mp4", text_tracks: mp4::text_tracks(bytes) }
    } else {
        ContainerProbe { format: "unknown", text_tracks: Vec::new() }
    }
}

mod mp4 {
    use super::TextTrack;

    const TEXT_HANDLERS: [&[u8; 4]; 4] = [b"sbtl", b"subt", b"text", b"clcp"];
    const TEXT_CODECS: [&[u8; 4]; 6] = [b"tx3g", b"wvtt", b"stpp", b"c608", b"c708", b"text"];

    /// Iterates `(type, payload)` over the boxes in `data`, stopping at the
    /// first malformed header.
    fn boxes(data: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
        let mut pos = 0usize;
        std::iter::from_fn(move || {
            if pos + 8 > data.len() {
                return None;
            }
            let size32 = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as u64;
            let kind: [u8; 4] = data[pos + 4..pos + 8].try_into().ok()?;
            let (header, size) = match size32 {
                0 => (8, (data.len() - pos) as u64),
                1 => {
                    let large = u64::from_be_bytes(data.get(pos + 8..pos + 16)?.try_into().ok()?);
                    (16, large)
                }
                s => (8, s),
            };
            let end = pos.checked_add(usize::try_from(size).ok()?)?;
            if size < header as u64 || end > data.len() {
                return None;
            }
            let payload = &data[pos + header..end];
            pos = end;
            Some((kind, payload))
        })
    }

    fn child<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
        boxes(data).find(|(k, _)| k == kind).map(|(_, p)| p)
    }

    pub fn text_tracks(bytes: &[u8]) -> Vec<TextTrack> {
        let Some(moov) = child(bytes, b"moov") else { return Vec::new() };
        boxes(moov)
            .filter(|(k, _)| k == b"trak")
            .filter_map(|(_, trak)| {
                let mdia = child(trak, b"mdia")?;
                let handler: [u8; 4] = child(mdia, b"hdlr")?.get(8..12)?.try_into().ok()?;
                let codec = child(mdia, b"minf")
                    .and_then(|m| child(m, b"stbl"))
                    .and_then(|s| child(s, b"stsd"))
                    .and_then(|stsd| stsd.get(12..16))
                    .and_then(|c| <[u8; 4]>::try_from(c).ok());
                let is_text = TEXT_HANDLERS.contains(&&handler) || codec.is_some_and(|c| TEXT_CODECS.contains(&&c));
                is_text.then(|| TextTrack {
                    codec: String::from_utf8_lossy(&codec.unwrap_or(handler)).into_owned(),
                    language: child(mdia, b"mdhd").and_then(mdhd_language),
                })
            })
            .collect()
    }

    fn mdhd_language(mdhd: &[u8]) -> Option<String> {
        let offset = if *mdhd.first()? == 1 { 32 } else { 20 };
        let packed = u16::from_be_bytes(mdhd.get(offset..offset + 2)?.try_into().ok()?);
        let lang: String = [10u16, 5, 0].iter().map(|s| (((packed >> s) & 0x1F) as u8 + 0x60) as char).collect();
        (lang.chars().all(|c| c.is_ascii_lowercase()) && lang != "und").then_some(lang)
    }
}

mod matroska {
    use super::TextTrack;

    const SEGMENT: u32 = 0x1853_8067;
    const TRACKS: u32 = 0x1654_AE6B;
    const TRACK_ENTRY: u32 = 0xAE;
    const TRACK_TYPE: u32 = 0x83;
    const CODEC_ID: u32 = 0x86;
    const LANGUAGE: u32 = 0x22_B59C;
    const LANGUAGE_BCP47: u32 = 0x22_B59D;
    const CLUSTER: u32 = 0x1F43_B675;
    const SUBTITLE_TRACK: u64 = 0x11;

    /// Reads an EBML variable-length integer. Returns (value, length, all_ones).
    fn vint(data: &[u8], keep_marker: bool) -> Option<(u64, usize, bool)> {
        let first = *data.first()?;
        let len = first.leading_zeros() as usize + 1;
        if len > 8 || data.len() < len {
            return None;
        }
        let mut value = if keep_marker { first as u64 } else { (first as u64) & (0xFF >> len) };
        for &b in &data[1..len] {
            value = (value << 8) | b as u64;
        }
        let all_ones = !keep_marker && value == (1u64 << (7 * len)) - 1;
        Some((value, len, all_ones))
    }

    /// Iterates `(id, payload)`; unknown-size elements extend to the end.
    fn elements(data: &[u8]) -> impl Iterator<Item = (u32, &[u8])> {
        let mut pos = 0usize;
        std::iter::from_fn(move || {
            let (id, id_len, _) = vint(data.get(pos..)?, true)?;
            let (size, size_len, unknown) = vint(data.get(pos + id_len..)?, false)?;
            let start = pos + id_len + size_len;
            let end = if unknown { data.len() } else { start.checked_add(usize::try_from(size).ok()?)?.min(data.len()) };
            if start > end {
                return None;
            }
            pos = end;
            Some((id as u32, &data[start..end]))
        })
    }

    pub fn text_tracks(bytes: &[u8]) -> Vec<TextTrack> {
        let mut out = Vec::new();
        for (id, segment) in elements(bytes) {
            if id != SEGMENT {
                continue;
            }
            for (id, tracks) in elements(segment) {
                if id == CLUSTER {
                    break; // Track headers precede media data.
                }
                if id != TRACKS {
                    continue;
                }
                for (id, entry) in elements(tracks) {
                    if id != TRACK_ENTRY {
                        continue;
                    }
                    let mut kind = 0u64;
                    let mut codec = String::new();
                    let mut language = None;
                    for (id, value) in elements(entry) {
                        match id {
                            TRACK_TYPE => kind = value.iter().fold(0u64, |acc, &b| (acc << 8) | b as u64),
                            CODEC_ID => codec = String::from_utf8_lossy(value).trim_end_matches('\0').to_string(),
                            LANGUAGE | LANGUAGE_BCP47 => {
                                language = Some(String::from_utf8_lossy(value).trim_end_matches('\0').to_string())
                            }
                            _ => {}
                        }
                    }
                    if kind == SUBTITLE_TRACK {
                        out.push(TextTrack { codec, language: language.filter(|l| l != "und") });
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut v = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(kind);
        v.extend_from_slice(payload);
        v
    }

    fn mp4_trak(handler: &[u8; 4], codec: &[u8; 4]) -> Vec<u8> {
        let mut hdlr = vec![0u8; 8];
        hdlr.extend_from_slice(handler);
        hdlr.extend_from_slice(&[0; 12]);
        let mut mdhd = vec![0u8; 20];
        // "eng" packed: e=5, n=14, g=7.
        mdhd.extend_from_slice(&((5u16 << 10) | (14 << 5) | 7).to_be_bytes());
        mdhd.extend_from_slice(&[0; 2]);
        let mut stsd = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd.extend(mp4_box(codec, &[0; 8]));
        let stbl = mp4_box(b"stbl", &mp4_box(b"stsd", &stsd));
        let minf = mp4_box(b"minf", &stbl);
        let mdia = mp4_box(b"mdia", &[mp4_box(b"mdhd", &mdhd), mp4_box(b"hdlr", &hdlr), minf].concat());
        mp4_box(b"trak", &mdia)
    }

    #[test]
    fn mp4_finds_subtitle_track_only() {
        let moov = mp4_box(b"moov", &[mp4_trak(b"vide", b"avc1"), mp4_trak(b"sbtl", b"tx3g")].concat());
        let file = [mp4_box(b"ftyp", b"isom\0\0\0\0"), moov].concat();
        let p = probe(&file);
        assert_eq!(p.format, "mp4");
        assert_eq!(p.text_tracks, vec![TextTrack { codec: "tx3g".into(), language: Some("eng".into()) }]);
    }

    #[test]
    fn mp4_without_text_tracks() {
        let file = [mp4_box(b"ftyp", b"isom\0\0\0\0"), mp4_box(b"moov", &mp4_trak(b"soun", b"mp4a"))].concat();
        assert!(probe(&file).text_tracks.is_empty());
    }

    #[test]
    fn garbage_is_unknown_and_does_not_panic() {
        assert_eq!(probe(b"hello world, not a video").format, "unknown");
        let mut truncated = mp4_box(b"ftyp", b"isom");
        truncated.extend_from_slice(&[0, 0, 0xFF, 0xFF, b'm', b'o', b'o', b'v', 1, 2]);
        assert!(probe(&truncated).text_tracks.is_empty());
    }
}
