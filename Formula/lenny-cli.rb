class LennyCli < Formula
  include Language::Python::Virtualenv

  desc "CLI for exploring Lenny's Podcast transcripts with fast + research routing"
  homepage "https://github.com/marcus-pm/lenny-cli"
  url "https://github.com/marcus-pm/lenny-cli.git",
      branch: "main",
      using: :git
  license "MIT"
  head "https://github.com/marcus-pm/lenny-cli.git", branch: "main"

  depends_on "python@3.12"
  depends_on "git"

  def install
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install buildpath
    bin.install_symlink libexec/"bin/lenny"
  end

  test do
    output = shell_output("#{bin}/lenny /quit 2>&1")
    assert_match "Lenny CLI", output
  end
end
