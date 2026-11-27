# spi-rs – a decoder for Samsung's SPI image format

This repository contains a crate called `spi`, which implements a decoder (but not an encoder) for
the SPI image format that is sometimes used by Samsung Notes. I created this for
[sdocx2pdf](https://github.com/squ1dd13/sdocx2pdf).

If you want to know how to use this crate, read the documentation in the comments in the source.
They are extensive.

I will write more about how this came to be at some point in the future when I have more time, but
it was something like this: I made a C++ harness that decoded SPI files by calling into the
libraries from the AArch64 Android version of Samsung Notes, and which compiled for AArch64 and ran
on my x86-64 machine through QEMU. This approach was inspired by [this
repo](https://github.com/flankerhqd/vendor-android-cves/tree/master/snotes-spi-fuzzing), which I
found via [this
presentation](https://i.blackhat.com/Asia-22/Friday-Materials/AS-22-He-The-Hidden-RCE-Surfaces-That-Control-The-Droids.pdf).
The function of interest from the libraries is `read_maetel_argb`, so I set up a dummy
`fake_read_maetel_argb` with the same signature but which did nothing. I set up an AGENTS.md file
with information about the format and libraries that I had found through manual reverse engineering
(locations of functions, structures I'd reconstituted, etc.) and details of the tools available on
the system. I then used OpenAI's Codex, mostly with GPT-5.6 Luna (but occasionally 5.6 Sol, for
parts that were trickier) to turn `fake_read_maetel_argb` into a full decoder. The main tools the
agents used were [radare2](https://github.com/radareorg/radare2) for static analysis and GDB for
dynamic analysis of `read_maetel_argb` (and for debugging `fake_read_maetel_argb`).

Using smaller models made the work quite difficult. The agents required constant supervision and
very clear tasks to complete, because otherwise they went completely off-track and made no
progress. It took two weeks to get C++ decoder that worked on all the sample SPI files I had
collected, and the incremental way it had been built meant that the project was an excellent
example of poor software design. I had a larger model (I forget which, but I will check and update
this) analyse the C++ decoder and draw up a plan for a Rust reimplementation. An explicit
instruction was that this was _not_ to be a direct Rust port; rather, it was to be a decoder
written in Rust that did the same thing as the C++ decoder. I used some smaller models to implement
the planned Rust version. The agents that did this were allowed access to the C++ decoder source,
but were not allowed to do any reverse-engineering work of their own. The Rust reimplementation was
completed in under 24 hours once I'd finished the plan.

This repo contains the Rust reimplementation, and has nothing from the original libraries or the
C++ decoder.
