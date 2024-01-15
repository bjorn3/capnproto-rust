use winnow::ascii::{digit0, digit1, hex_uint, newline, oct_digit0};
use winnow::combinator::{alt, delimited, eof, not, opt, peek, preceded, repeat, separated};
use winnow::stream::AsChar;
use winnow::token::{one_of, tag, take, take_while};
use winnow::{Located, PResult, Parser};

// ================ kj/parse/char.h ================

// =======================================================================================
// Basic character classes.

fn is_name_start(input: char) -> bool {
    input.is_alphabetic() || input == '_'
}

fn is_name_char(input: char) -> bool {
    input.is_alphanumeric() || input == '_'
}

fn discard_whitespace(input: &mut Located<&str>) -> PResult<()> {
    repeat(0.., one_of([' ', '\x0C', '\n', '\r', '\t', '\x0B'])).parse_next(input)
}

// =======================================================================================
// Identifiers

fn parse_identifier(input: &mut Located<&str>) -> PResult<String> {
    (one_of(is_name_start), take_while(0.., is_name_char))
        .recognize()
        .map(|s: &str| s.to_owned())
        .parse_next(input)
}

// =======================================================================================
// Integers

fn parse_integer(input: &mut Located<&str>) -> PResult<u64> {
    (
        alt((
            (tag("0x"), hex_uint).map(|(_, u)| u),
            (tag("0"), oct_digit0)
                .recognize()
                .try_map(|s| u64::from_str_radix(s, 8)),
            (
                one_of(['1', '2', '3', '4', '5', '6', '7', '8', '9']),
                digit0,
            )
                .recognize()
                .try_map(|s: &str| s.parse()),
        )),
        peek(not(one_of(|c: char| c.is_alpha() || c == '_' || c == '.'))),
    )
        .map(|(i, ())| i)
        .parse_next(input)
}

// =======================================================================================
// Numbers (i.e. floats)

fn parse_number(input: &mut Located<&str>) -> PResult<f64> {
    (
        (
            digit1,
            opt((tag('.'), digit0)),
            opt((one_of(['e', 'E']), opt(one_of(['+', '-'])), digit0)),
        ),
        peek(not(one_of(|c: char| c.is_alpha() || c == '_' || c == '.'))),
    )
        .recognize()
        .try_map(|s: &str| s.parse())
        .parse_next(input)
}

// =======================================================================================
// Quoted strings

fn parse_escape_sequence(input: &mut Located<&str>) -> PResult<char> {
    alt((
        "a".value('\x07'),
        "b".value('\x08'),
        "f".value('\x0C'),
        "n".value('\n'),
        "r".value('\r'),
        "t".value('\t'),
        "v".value('\x0B'),
        preceded(
            "x",
            take(2u32).and_then(hex_uint).map(|x: u8| {
                assert!(x < 128);
                char::from_u32(x.into()).unwrap()
            }),
        ),
    ))
    .parse_next(input)
}

fn parse_double_quoted_string(input: &mut Located<&str>) -> PResult<String> {
    delimited(
        tag('"'),
        repeat(
            0..,
            alt((
                one_of(|c| c != '\\' && c != '\n' && c != '"'),
                (tag('\\'), parse_escape_sequence).map(|(_, c)| c),
            )),
        ),
        tag('"'),
    )
    .parse_next(input)
}

fn parse_double_quoted_hex_binary(input: &mut Located<&str>) -> PResult<Vec<u8>> {
    delimited(
        tag("0x\""),
        (
            repeat(
                1..,
                (discard_whitespace, take(2u32).and_then(hex_uint)).map(|((), val): ((), u8)| val),
            ),
            discard_whitespace,
        )
            .map(|(bytes, ())| bytes),
        tag('"'),
    )
    .parse_next(input)
}

// ================ capnp/compiler/lexer.capnp ================

pub struct Token {
    pub start_byte: u32,
    pub end_byte: u32,
    pub kind: TokenKind,
}

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind.fmt(f)
    }
}

#[derive(Debug)]
pub enum TokenKind {
    Identifier(String),
    StringLiteral(String),
    BinaryLiteral(Vec<u8>),
    IntegerLiteral(u64),
    FloatLiteral(f64),
    Operator(String),
    ParenthesizedList(Vec<Vec<Token>>),
    BracketedList(Vec<Vec<Token>>),
}

pub struct Statement {
    pub tokens: Vec<Token>,
    pub kind: StatementKind,
    pub doc_comment: Option<String>,
    pub start_byte: u32,
    pub end_byte: u32,
}

impl std::fmt::Debug for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Statement")
            .field("doc_comment", &self.doc_comment)
            .field("tokens", &self.tokens)
            .field("kind", &self.kind)
            .finish()
    }
}

#[derive(Debug)]
pub enum StatementKind {
    Line,
    Block(Vec<Statement>),
}

// ================ capnp/compiler/lexer.c++ ================

fn parse_discard_comment(input: &mut Located<&str>) -> PResult<()> {
    (
        tag('#'),
        repeat(0.., one_of(|c| c != '\n')).map(|()| ()),
        alt((tag('\n'), eof)),
    )
        .void()
        .parse_next(input)
}

fn parse_save_comment(input: &mut Located<&str>) -> PResult<String> {
    (
        tag('#'),
        opt(tag(' ')),
        repeat(0.., one_of(|c| c != '\n')),
        alt((tag('\n'), eof)),
    )
        .map(|(_, _, s, _)| s)
        .parse_next(input)
}

fn parse_utf8_bom(input: &mut Located<&str>) -> PResult<()> {
    tag('\u{feff}').void().parse_next(input)
}

fn parse_boms_and_whitespace(input: &mut Located<&str>) -> PResult<()> {
    (
        discard_whitespace,
        repeat(0.., (parse_utf8_bom, discard_whitespace)).map(|()| ()),
    )
        .void()
        .parse_next(input)
}

fn parse_comments_and_whitespace(input: &mut Located<&str>) -> PResult<()> {
    (
        parse_boms_and_whitespace,
        repeat(0.., (parse_discard_comment, parse_boms_and_whitespace)).map(|()| ()),
    )
        .void()
        .parse_next(input)
}

fn parse_discard_line_whitespace(input: &mut Located<&str>) -> PResult<()> {
    repeat(0.., one_of([' ', '\x0C', '\t', '\x0B'])).parse_next(input)
}

fn parse_doc_comment(input: &mut Located<&str>) -> PResult<Option<String>> {
    opt((
        parse_discard_line_whitespace,
        opt(newline),
        repeat(
            1..,
            (parse_discard_line_whitespace, parse_save_comment).map(|((), s)| s),
        ),
    )
        .map(|((), _, comment): ((), _, Vec<String>)| comment.join("\n")))
    .parse_next(input)
}

fn parse_comma_delimited_list(input: &mut Located<&str>) -> PResult<Vec<Vec<Token>>> {
    separated(0.., parse_token_sequence, tag(','))
        .map(|mut token_lists: Vec<_>| {
            if token_lists.last().map_or(false, |tokens| tokens.is_empty()) {
                token_lists.pop();
            }
            token_lists
        })
        .parse_next(input)
}

fn parse_token(input: &mut Located<&str>) -> PResult<Token> {
    let (kind, range) = parse_token_inner.with_span().parse_next(input)?;
    Ok(Token {
        start_byte: range.start as u32,
        end_byte: range.end as u32,
        kind,
    })
}

fn parse_token_inner(input: &mut Located<&str>) -> PResult<TokenKind> {
    alt((
        parse_identifier.map(TokenKind::Identifier),
        parse_double_quoted_string.map(TokenKind::StringLiteral),
        delimited('`', repeat(0.., one_of(|c| c != '\r' && c != '\n')), '`')
            .map(TokenKind::StringLiteral),
        parse_double_quoted_hex_binary.map(TokenKind::BinaryLiteral),
        parse_integer.map(TokenKind::IntegerLiteral),
        parse_number.map(TokenKind::FloatLiteral),
        repeat(1.., one_of(|c| "!$%&*+-./:<=>?@^|~".contains(c))).map(|s| TokenKind::Operator(s)),
        delimited('(', parse_comma_delimited_list, ')').map(TokenKind::ParenthesizedList),
        delimited('[', parse_comma_delimited_list, ']').map(TokenKind::BracketedList),
    ))
    .parse_next(input)
}

fn parse_token_sequence(input: &mut Located<&str>) -> PResult<Vec<Token>> {
    (
        parse_comments_and_whitespace,
        repeat(
            0..,
            (parse_token, parse_comments_and_whitespace).map(|(token, ())| token),
        ),
    )
        .map(|((), tokens)| tokens)
        .parse_next(input)
}

fn parse_statement_end(input: &mut Located<&str>) -> PResult<Statement> {
    alt((
        (tag(';'), parse_doc_comment).map(|(_, doc_comment)| Statement {
            tokens: vec![],
            kind: StatementKind::Line,
            doc_comment,
            start_byte: 0,
            end_byte: 0,
        }),
        (
            tag('{'),
            parse_doc_comment,
            parse_statement_sequence,
            tag('}'),
            parse_doc_comment,
        )
            .map(
                |(_, doc_comment, statements, _, late_doc_comment)| Statement {
                    tokens: vec![],
                    kind: StatementKind::Block(statements),
                    doc_comment: doc_comment.or(late_doc_comment),
                    start_byte: 0,
                    end_byte: 0,
                },
            ),
    ))
    .parse_next(input)
}

fn parse_statement(input: &mut Located<&str>) -> PResult<Statement> {
    (parse_token_sequence, parse_statement_end)
        .with_span()
        .map(|((tokens, mut statement), range)| {
            statement.tokens = tokens;
            statement.start_byte = range.start as u32;
            statement.end_byte = range.end as u32;
            statement
        })
        .parse_next(input)
}

pub fn parse_statement_sequence(input: &mut Located<&str>) -> PResult<Vec<Statement>> {
    (
        parse_comments_and_whitespace,
        repeat(
            0..,
            (parse_statement, parse_comments_and_whitespace).map(|(statement, ())| statement),
        ),
    )
        .map(|((), statements)| statements)
        .parse_next(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculator_example() {
        let capnp = r#"
        # Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors
        # Licensed under the MIT License:
        #
        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:
        #
        # The above copyright notice and this permission notice shall be included in
        # all copies or substantial portions of the Software.
        #
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        # THE SOFTWARE.

        @0x85150b117366d14b;

        interface Calculator {
          # A "simple" mathematical calculator, callable via RPC.
          #
          # But, to show off Cap'n Proto, we add some twists:
          #
          # - You can use the result from one call as the input to the next
          #   without a network round trip.  To accomplish this, evaluate()
          #   returns a `Value` object wrapping the actual numeric value.
          #   This object may be used in a subsequent expression.  With
          #   promise pipelining, the Value can actually be used before
          #   the evaluate() call that creates it returns!
          #
          # - You can define new functions, and then call them.  This again
          #   shows off pipelining, but it also gives the client the
          #   opportunity to define a function on the client side and have
          #   the server call back to it.
          #
          # - The basic arithmetic operators are exposed as Functions, and
          #   you have to call getOperator() to obtain them from the server.
          #   This again demonstrates pipelining -- using getOperator() to
          #   get each operator and then using them in evaluate() still
          #   only takes one network round trip.

          evaluate @0 (expression :Expression) -> (value :Value);
          # Evaluate the given expression and return the result.  The
          # result is returned wrapped in a Value interface so that you
          # may pass it back to the server in a pipelined request.  To
          # actually get the numeric value, you must call read() on the
          # Value -- but again, this can be pipelined so that it incurs
          # no additional latency.

          struct Expression {
            # A numeric expression.

            union {
              literal @0 :Float64;
              # A literal numeric value.

              previousResult @1 :Value;
              # A value that was (or, will be) returned by a previous
              # evaluate().

              parameter @2 :UInt32;
              # A parameter to the function (only valid in function bodies;
              # see defFunction).

              call :group {
                # Call a function on a list of parameters.
                function @3 :Function;
                params @4 :List(Expression);
              }
            }
          }

          interface Value {
            # Wraps a numeric value in an RPC object.  This allows the value
            # to be used in subsequent evaluate() requests without the client
            # waiting for the evaluate() that returns the Value to finish.

            read @0 () -> (value :Float64);
            # Read back the raw numeric value.
          }

          defFunction @1 (paramCount :Int32, body :Expression)
                      -> (func :Function);
          # Define a function that takes `paramCount` parameters and returns the
          # evaluation of `body` after substituting these parameters.

          interface Function {
            # An algebraic function.  Can be called directly, or can be used inside
            # an Expression.
            #
            # A client can create a Function that runs on the server side using
            # `defFunction()` or `getOperator()`.  Alternatively, a client can
            # implement a Function on the client side and the server will call back
            # to it.  However, a function defined on the client side will require a
            # network round trip whenever the server needs to call it, whereas
            # functions defined on the server and then passed back to it are called
            # locally.

            call @0 (params :List(Float64)) -> (value :Float64);
            # Call the function on the given parameters.
          }

          getOperator @2 (op :Operator) -> (func :Function);
          # Get a Function representing an arithmetic operator, which can then be
          # used in Expressions.

          enum Operator {
            add @0;
            subtract @1;
            multiply @2;
            divide @3;
          }
        }
        "#;

        expect_test::expect![[r#"
            Ok(
                [
                    Statement {
                        doc_comment: None,
                        tokens: [
                            Operator(
                                "@",
                            ),
                            IntegerLiteral(
                                9589583151133806923,
                            ),
                        ],
                        kind: Line,
                    },
                    Statement {
                        doc_comment: Some(
                            "A \"simple\" mathematical calculator, callable via RPC.\n\nBut, to show off Cap'n Proto, we add some twists:\n\n- You can use the result from one call as the input to the next\n  without a network round trip.  To accomplish this, evaluate()\n  returns a `Value` object wrapping the actual numeric value.\n  This object may be used in a subsequent expression.  With\n  promise pipelining, the Value can actually be used before\n  the evaluate() call that creates it returns!\n\n- You can define new functions, and then call them.  This again\n  shows off pipelining, but it also gives the client the\n  opportunity to define a function on the client side and have\n  the server call back to it.\n\n- The basic arithmetic operators are exposed as Functions, and\n  you have to call getOperator() to obtain them from the server.\n  This again demonstrates pipelining -- using getOperator() to\n  get each operator and then using them in evaluate() still\n  only takes one network round trip.",
                        ),
                        tokens: [
                            Identifier(
                                "interface",
                            ),
                            Identifier(
                                "Calculator",
                            ),
                        ],
                        kind: Block(
                            [
                                Statement {
                                    doc_comment: Some(
                                        "Evaluate the given expression and return the result.  The\nresult is returned wrapped in a Value interface so that you\nmay pass it back to the server in a pipelined request.  To\nactually get the numeric value, you must call read() on the\nValue -- but again, this can be pipelined so that it incurs\nno additional latency.",
                                    ),
                                    tokens: [
                                        Identifier(
                                            "evaluate",
                                        ),
                                        Operator(
                                            "@",
                                        ),
                                        IntegerLiteral(
                                            0,
                                        ),
                                        ParenthesizedList(
                                            [
                                                [
                                                    Identifier(
                                                        "expression",
                                                    ),
                                                    Operator(
                                                        ":",
                                                    ),
                                                    Identifier(
                                                        "Expression",
                                                    ),
                                                ],
                                            ],
                                        ),
                                        Operator(
                                            "->",
                                        ),
                                        ParenthesizedList(
                                            [
                                                [
                                                    Identifier(
                                                        "value",
                                                    ),
                                                    Operator(
                                                        ":",
                                                    ),
                                                    Identifier(
                                                        "Value",
                                                    ),
                                                ],
                                            ],
                                        ),
                                    ],
                                    kind: Line,
                                },
                                Statement {
                                    doc_comment: Some(
                                        "A numeric expression.",
                                    ),
                                    tokens: [
                                        Identifier(
                                            "struct",
                                        ),
                                        Identifier(
                                            "Expression",
                                        ),
                                    ],
                                    kind: Block(
                                        [
                                            Statement {
                                                doc_comment: None,
                                                tokens: [
                                                    Identifier(
                                                        "union",
                                                    ),
                                                ],
                                                kind: Block(
                                                    [
                                                        Statement {
                                                            doc_comment: Some(
                                                                "A literal numeric value.",
                                                            ),
                                                            tokens: [
                                                                Identifier(
                                                                    "literal",
                                                                ),
                                                                Operator(
                                                                    "@",
                                                                ),
                                                                IntegerLiteral(
                                                                    0,
                                                                ),
                                                                Operator(
                                                                    ":",
                                                                ),
                                                                Identifier(
                                                                    "Float64",
                                                                ),
                                                            ],
                                                            kind: Line,
                                                        },
                                                        Statement {
                                                            doc_comment: Some(
                                                                "A value that was (or, will be) returned by a previous\nevaluate().",
                                                            ),
                                                            tokens: [
                                                                Identifier(
                                                                    "previousResult",
                                                                ),
                                                                Operator(
                                                                    "@",
                                                                ),
                                                                IntegerLiteral(
                                                                    1,
                                                                ),
                                                                Operator(
                                                                    ":",
                                                                ),
                                                                Identifier(
                                                                    "Value",
                                                                ),
                                                            ],
                                                            kind: Line,
                                                        },
                                                        Statement {
                                                            doc_comment: Some(
                                                                "A parameter to the function (only valid in function bodies;\nsee defFunction).",
                                                            ),
                                                            tokens: [
                                                                Identifier(
                                                                    "parameter",
                                                                ),
                                                                Operator(
                                                                    "@",
                                                                ),
                                                                IntegerLiteral(
                                                                    2,
                                                                ),
                                                                Operator(
                                                                    ":",
                                                                ),
                                                                Identifier(
                                                                    "UInt32",
                                                                ),
                                                            ],
                                                            kind: Line,
                                                        },
                                                        Statement {
                                                            doc_comment: Some(
                                                                "Call a function on a list of parameters.",
                                                            ),
                                                            tokens: [
                                                                Identifier(
                                                                    "call",
                                                                ),
                                                                Operator(
                                                                    ":",
                                                                ),
                                                                Identifier(
                                                                    "group",
                                                                ),
                                                            ],
                                                            kind: Block(
                                                                [
                                                                    Statement {
                                                                        doc_comment: None,
                                                                        tokens: [
                                                                            Identifier(
                                                                                "function",
                                                                            ),
                                                                            Operator(
                                                                                "@",
                                                                            ),
                                                                            IntegerLiteral(
                                                                                3,
                                                                            ),
                                                                            Operator(
                                                                                ":",
                                                                            ),
                                                                            Identifier(
                                                                                "Function",
                                                                            ),
                                                                        ],
                                                                        kind: Line,
                                                                    },
                                                                    Statement {
                                                                        doc_comment: None,
                                                                        tokens: [
                                                                            Identifier(
                                                                                "params",
                                                                            ),
                                                                            Operator(
                                                                                "@",
                                                                            ),
                                                                            IntegerLiteral(
                                                                                4,
                                                                            ),
                                                                            Operator(
                                                                                ":",
                                                                            ),
                                                                            Identifier(
                                                                                "List",
                                                                            ),
                                                                            ParenthesizedList(
                                                                                [
                                                                                    [
                                                                                        Identifier(
                                                                                            "Expression",
                                                                                        ),
                                                                                    ],
                                                                                ],
                                                                            ),
                                                                        ],
                                                                        kind: Line,
                                                                    },
                                                                ],
                                                            ),
                                                        },
                                                    ],
                                                ),
                                            },
                                        ],
                                    ),
                                },
                                Statement {
                                    doc_comment: Some(
                                        "Wraps a numeric value in an RPC object.  This allows the value\nto be used in subsequent evaluate() requests without the client\nwaiting for the evaluate() that returns the Value to finish.",
                                    ),
                                    tokens: [
                                        Identifier(
                                            "interface",
                                        ),
                                        Identifier(
                                            "Value",
                                        ),
                                    ],
                                    kind: Block(
                                        [
                                            Statement {
                                                doc_comment: Some(
                                                    "Read back the raw numeric value.",
                                                ),
                                                tokens: [
                                                    Identifier(
                                                        "read",
                                                    ),
                                                    Operator(
                                                        "@",
                                                    ),
                                                    IntegerLiteral(
                                                        0,
                                                    ),
                                                    ParenthesizedList(
                                                        [],
                                                    ),
                                                    Operator(
                                                        "->",
                                                    ),
                                                    ParenthesizedList(
                                                        [
                                                            [
                                                                Identifier(
                                                                    "value",
                                                                ),
                                                                Operator(
                                                                    ":",
                                                                ),
                                                                Identifier(
                                                                    "Float64",
                                                                ),
                                                            ],
                                                        ],
                                                    ),
                                                ],
                                                kind: Line,
                                            },
                                        ],
                                    ),
                                },
                                Statement {
                                    doc_comment: Some(
                                        "Define a function that takes `paramCount` parameters and returns the\nevaluation of `body` after substituting these parameters.",
                                    ),
                                    tokens: [
                                        Identifier(
                                            "defFunction",
                                        ),
                                        Operator(
                                            "@",
                                        ),
                                        IntegerLiteral(
                                            1,
                                        ),
                                        ParenthesizedList(
                                            [
                                                [
                                                    Identifier(
                                                        "paramCount",
                                                    ),
                                                    Operator(
                                                        ":",
                                                    ),
                                                    Identifier(
                                                        "Int32",
                                                    ),
                                                ],
                                                [
                                                    Identifier(
                                                        "body",
                                                    ),
                                                    Operator(
                                                        ":",
                                                    ),
                                                    Identifier(
                                                        "Expression",
                                                    ),
                                                ],
                                            ],
                                        ),
                                        Operator(
                                            "->",
                                        ),
                                        ParenthesizedList(
                                            [
                                                [
                                                    Identifier(
                                                        "func",
                                                    ),
                                                    Operator(
                                                        ":",
                                                    ),
                                                    Identifier(
                                                        "Function",
                                                    ),
                                                ],
                                            ],
                                        ),
                                    ],
                                    kind: Line,
                                },
                                Statement {
                                    doc_comment: Some(
                                        "An algebraic function.  Can be called directly, or can be used inside\nan Expression.\n\nA client can create a Function that runs on the server side using\n`defFunction()` or `getOperator()`.  Alternatively, a client can\nimplement a Function on the client side and the server will call back\nto it.  However, a function defined on the client side will require a\nnetwork round trip whenever the server needs to call it, whereas\nfunctions defined on the server and then passed back to it are called\nlocally.",
                                    ),
                                    tokens: [
                                        Identifier(
                                            "interface",
                                        ),
                                        Identifier(
                                            "Function",
                                        ),
                                    ],
                                    kind: Block(
                                        [
                                            Statement {
                                                doc_comment: Some(
                                                    "Call the function on the given parameters.",
                                                ),
                                                tokens: [
                                                    Identifier(
                                                        "call",
                                                    ),
                                                    Operator(
                                                        "@",
                                                    ),
                                                    IntegerLiteral(
                                                        0,
                                                    ),
                                                    ParenthesizedList(
                                                        [
                                                            [
                                                                Identifier(
                                                                    "params",
                                                                ),
                                                                Operator(
                                                                    ":",
                                                                ),
                                                                Identifier(
                                                                    "List",
                                                                ),
                                                                ParenthesizedList(
                                                                    [
                                                                        [
                                                                            Identifier(
                                                                                "Float64",
                                                                            ),
                                                                        ],
                                                                    ],
                                                                ),
                                                            ],
                                                        ],
                                                    ),
                                                    Operator(
                                                        "->",
                                                    ),
                                                    ParenthesizedList(
                                                        [
                                                            [
                                                                Identifier(
                                                                    "value",
                                                                ),
                                                                Operator(
                                                                    ":",
                                                                ),
                                                                Identifier(
                                                                    "Float64",
                                                                ),
                                                            ],
                                                        ],
                                                    ),
                                                ],
                                                kind: Line,
                                            },
                                        ],
                                    ),
                                },
                                Statement {
                                    doc_comment: Some(
                                        "Get a Function representing an arithmetic operator, which can then be\nused in Expressions.",
                                    ),
                                    tokens: [
                                        Identifier(
                                            "getOperator",
                                        ),
                                        Operator(
                                            "@",
                                        ),
                                        IntegerLiteral(
                                            2,
                                        ),
                                        ParenthesizedList(
                                            [
                                                [
                                                    Identifier(
                                                        "op",
                                                    ),
                                                    Operator(
                                                        ":",
                                                    ),
                                                    Identifier(
                                                        "Operator",
                                                    ),
                                                ],
                                            ],
                                        ),
                                        Operator(
                                            "->",
                                        ),
                                        ParenthesizedList(
                                            [
                                                [
                                                    Identifier(
                                                        "func",
                                                    ),
                                                    Operator(
                                                        ":",
                                                    ),
                                                    Identifier(
                                                        "Function",
                                                    ),
                                                ],
                                            ],
                                        ),
                                    ],
                                    kind: Line,
                                },
                                Statement {
                                    doc_comment: None,
                                    tokens: [
                                        Identifier(
                                            "enum",
                                        ),
                                        Identifier(
                                            "Operator",
                                        ),
                                    ],
                                    kind: Block(
                                        [
                                            Statement {
                                                doc_comment: None,
                                                tokens: [
                                                    Identifier(
                                                        "add",
                                                    ),
                                                    Operator(
                                                        "@",
                                                    ),
                                                    IntegerLiteral(
                                                        0,
                                                    ),
                                                ],
                                                kind: Line,
                                            },
                                            Statement {
                                                doc_comment: None,
                                                tokens: [
                                                    Identifier(
                                                        "subtract",
                                                    ),
                                                    Operator(
                                                        "@",
                                                    ),
                                                    IntegerLiteral(
                                                        1,
                                                    ),
                                                ],
                                                kind: Line,
                                            },
                                            Statement {
                                                doc_comment: None,
                                                tokens: [
                                                    Identifier(
                                                        "multiply",
                                                    ),
                                                    Operator(
                                                        "@",
                                                    ),
                                                    IntegerLiteral(
                                                        2,
                                                    ),
                                                ],
                                                kind: Line,
                                            },
                                            Statement {
                                                doc_comment: None,
                                                tokens: [
                                                    Identifier(
                                                        "divide",
                                                    ),
                                                    Operator(
                                                        "@",
                                                    ),
                                                    IntegerLiteral(
                                                        3,
                                                    ),
                                                ],
                                                kind: Line,
                                            },
                                        ],
                                    ),
                                },
                            ],
                        ),
                    },
                ],
            )"#]].assert_eq(&format!(
            "{:#?}",
            parse_statement_sequence.parse(Located::new(capnp))
        ));
    }
}
