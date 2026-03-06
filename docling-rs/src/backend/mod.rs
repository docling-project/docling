pub mod asciidoc;
pub mod csv;
pub mod docx;
pub mod html;
pub mod image;
pub mod jats;
pub mod json_docling;
pub mod latex;
pub mod markdown;
pub mod mets_gbs;
pub mod pdf;
pub mod pptx;
pub mod uspto;
pub mod webvtt;
pub mod xbrl;
pub mod xlsx;

use std::path::Path;

use crate::models::document::DoclingDocument;

pub trait Backend: Send + Sync {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument>;
}

pub fn resolve_common_entities(content: &mut String) {
    let entities = [
        ("&lsqb;", "["),
        ("&rsqb;", "]"),
        ("&lsquo;", "\u{2018}"),
        ("&rsquo;", "\u{2019}"),
        ("&ldquo;", "\u{201C}"),
        ("&rdquo;", "\u{201D}"),
        ("&ndash;", "\u{2013}"),
        ("&mdash;", "\u{2014}"),
        ("&hellip;", "\u{2026}"),
        ("&trade;", "\u{2122}"),
        ("&reg;", "\u{00AE}"),
        ("&copy;", "\u{00A9}"),
        ("&times;", "\u{00D7}"),
        ("&divide;", "\u{00F7}"),
        ("&minus;", "\u{2212}"),
        ("&plusmn;", "\u{00B1}"),
        ("&deg;", "\u{00B0}"),
        ("&micro;", "\u{00B5}"),
        ("&bull;", "\u{2022}"),
        ("&middot;", "\u{00B7}"),
        ("&sect;", "\u{00A7}"),
        ("&para;", "\u{00B6}"),
        ("&dagger;", "\u{2020}"),
        ("&Dagger;", "\u{2021}"),
        ("&alpha;", "\u{03B1}"),
        ("&beta;", "\u{03B2}"),
        ("&gamma;", "\u{03B3}"),
        ("&delta;", "\u{03B4}"),
        ("&epsilon;", "\u{03B5}"),
        ("&zeta;", "\u{03B6}"),
        ("&eta;", "\u{03B7}"),
        ("&theta;", "\u{03B8}"),
        ("&iota;", "\u{03B9}"),
        ("&kappa;", "\u{03BA}"),
        ("&lambda;", "\u{03BB}"),
        ("&mu;", "\u{03BC}"),
        ("&nu;", "\u{03BD}"),
        ("&xi;", "\u{03BE}"),
        ("&pi;", "\u{03C0}"),
        ("&rho;", "\u{03C1}"),
        ("&sigma;", "\u{03C3}"),
        ("&tau;", "\u{03C4}"),
        ("&upsilon;", "\u{03C5}"),
        ("&phi;", "\u{03C6}"),
        ("&chi;", "\u{03C7}"),
        ("&psi;", "\u{03C8}"),
        ("&omega;", "\u{03C9}"),
        ("&ohgr;", "\u{03C9}"),
        ("&agr;", "\u{03B1}"),
        ("&bgr;", "\u{03B2}"),
        ("&ggr;", "\u{03B3}"),
        ("&dgr;", "\u{03B4}"),
        ("&egr;", "\u{03B5}"),
        ("&zgr;", "\u{03B6}"),
        ("&eegr;", "\u{03B7}"),
        ("&thgr;", "\u{03B8}"),
        ("&igr;", "\u{03B9}"),
        ("&kgr;", "\u{03BA}"),
        ("&lgr;", "\u{03BB}"),
        ("&mgr;", "\u{03BC}"),
        ("&ngr;", "\u{03BD}"),
        ("&xgr;", "\u{03BE}"),
        ("&pgr;", "\u{03C0}"),
        ("&rgr;", "\u{03C1}"),
        ("&sgr;", "\u{03C3}"),
        ("&tgr;", "\u{03C4}"),
        ("&ugr;", "\u{03C5}"),
        ("&phgr;", "\u{03C6}"),
        ("&khgr;", "\u{03C7}"),
        ("&psgr;", "\u{03C8}"),
        ("&Agr;", "\u{0391}"),
        ("&Bgr;", "\u{0392}"),
        ("&Ggr;", "\u{0393}"),
        ("&Dgr;", "\u{0394}"),
        ("&EEgr;", "\u{0397}"),
        ("&THgr;", "\u{0398}"),
        ("&Lgr;", "\u{039B}"),
        ("&Xgr;", "\u{039E}"),
        ("&Pgr;", "\u{03A0}"),
        ("&Sgr;", "\u{03A3}"),
        ("&PHgr;", "\u{03A6}"),
        ("&PSgr;", "\u{03A8}"),
        ("&OHgr;", "\u{03A9}"),
        ("&le;", "\u{2264}"),
        ("&ge;", "\u{2265}"),
        ("&ne;", "\u{2260}"),
        ("&infin;", "\u{221E}"),
        ("&sum;", "\u{2211}"),
        ("&prod;", "\u{220F}"),
        ("&radic;", "\u{221A}"),
        ("&prop;", "\u{221D}"),
        ("&larr;", "\u{2190}"),
        ("&rarr;", "\u{2192}"),
        ("&uarr;", "\u{2191}"),
        ("&darr;", "\u{2193}"),
        ("&harr;", "\u{2194}"),
        ("&frac12;", "\u{00BD}"),
        ("&frac14;", "\u{00BC}"),
        ("&frac34;", "\u{00BE}"),
        ("&nbsp;", "\u{00A0}"),
        ("&emsp;", "\u{2003}"),
        ("&ensp;", "\u{2002}"),
        ("&thinsp;", "\u{2009}"),
        ("&equals;", "="),
        ("&plus;", "+"),
        ("&comma;", ","),
        ("&period;", "."),
        ("&colon;", ":"),
        ("&semi;", ";"),
        ("&excl;", "!"),
        ("&quest;", "?"),
        ("&num;", "#"),
        ("&percnt;", "%"),
        ("&dollar;", "$"),
        ("&commat;", "@"),
        ("&ast;", "*"),
        ("&sol;", "/"),
        ("&bsol;", "\\"),
        ("&verbar;", "|"),
        ("&hyphen;", "-"),
        ("&lowbar;", "_"),
        ("&lpar;", "("),
        ("&rpar;", ")"),
        ("&lbrace;", "{"),
        ("&rbrace;", "}"),
        ("&circ;", "\u{02C6}"),
        ("&tilde;", "\u{02DC}"),
        ("&grave;", "`"),
        ("&af;", "\u{2061}"),
        ("&it;", "\u{2062}"),
        ("&ic;", "\u{2063}"),
        ("&ii;", "\u{2064}"),
        ("&ApplyFunction;", "\u{2061}"),
        ("&InvisibleTimes;", "\u{2062}"),
    ];
    for (entity, replacement) in &entities {
        if content.contains(entity) {
            *content = content.replace(entity, replacement);
        }
    }
    strip_unknown_entities(content);
}

fn strip_unknown_entities(content: &mut String) {
    use std::fmt::Write;
    let mut result = String::with_capacity(content.len());
    let mut chars = content.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '&' {
            let mut entity = String::from('&');
            let mut is_entity = false;
            while let Some(&next) = chars.peek() {
                entity.push(next);
                chars.next();
                if next == ';' {
                    is_entity = true;
                    break;
                }
                if next.is_whitespace() || next == '<' || entity.len() > 32 {
                    break;
                }
            }
            if is_entity
                && !entity.starts_with("&#")
                && entity != "&amp;"
                && entity != "&lt;"
                && entity != "&gt;"
                && entity != "&quot;"
                && entity != "&apos;"
            {
                let inner = &entity[1..entity.len() - 1];
                let _ = write!(result, "[{}]", inner);
            } else {
                result.push_str(&entity);
            }
        } else {
            result.push(ch);
        }
    }
    *content = result;
}
