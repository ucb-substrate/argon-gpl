use std::fmt::Write;

use anyhow::{bail, Context};
use lrlex::lrlex_mod;
use lrpar::lrpar_mod;

lrlex_mod!("cadlang.l");
lrpar_mod!("cadlang.y");

pub use cadlang_y::*;

pub struct CadlangAst<'a> {
    pub decls: Vec<Decl<'a>>,
}

pub fn parse(input: &str) -> Result<CadlangAst<'_>, anyhow::Error> {
    // Get the `LexerDef` for the `cadlang` language.
    let lexerdef = cadlang_l::lexerdef();
    // Now we create a lexer with the `lexer` method with which
    // we can lex an input.
    let lexer = lexerdef.lexer(input);
    // Pass the lexer to the parser and lex and parse the input.
    let (res, errs) = cadlang_y::parse(&lexer);
    if !errs.is_empty() {
        let mut err = String::new();
        for e in errs {
            write!(&mut err, "{}", e.pp(&lexer, &cadlang_y::token_epp))
                .with_context(|| "failed to write to string buffer")?;
        }
        bail!("{err}");
    }
    match res {
        Some(Ok(decls)) => Ok(CadlangAst { decls }),
        _ => bail!("Unable to evaluate expression."),
    }
}
