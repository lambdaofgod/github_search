defmodule GithubSearch.RawFileDownloader do
  def load_readme(user_name, repo) do
    url = "https://raw.githubusercontent.com/#{user_name}/#{repo}/master/README.md"

    try do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} = HTTPoison.get(url)
      body
      body
    rescue
      _ -> nil
    end
  end

  def load_readme(repo_name_with_user) do
    case repo_name_with_user |> String.split("/") do
      [user_name, repo_name] -> load_readme(user_name, repo_name)
      _ -> nil
    end
  end

  def load_readmes(repo_names) do
    readmes = repo_names |> Task.async_stream(&load_readme/1)

    for {repo_name, {:ok, readme}} <- repo_names |> Enum.zip(readmes) do
      %{"repo" => repo_name, "readme" => readme}
    end
  end
end

defmodule GithubSearch.ReadmeDownloader do
  require Logger

  def from_repos_csv(repos_csv_path, output_path) do
    repos = repos_csv_path |> load_repos_from_csv()
    Logger.info("Loaded #{Enum.count(repos)} repos from #{repos_csv_path}")
    repos_with_readmes = repos |> GithubSearch.RawFileDownloader.load_readmes()
    Logger.info("Downloaded readmes")
    write_json(output_path, repos_with_readmes)
  end

  def load_repos_from_csv(repos_csv_path) do
    repos_csv_path
    |> File.stream!()
    |> CSV.decode(separator: ?,)
    |> Stream.drop(1)
    |> Enum.map(fn {:ok, [repo | _]} -> repo end)
  end

  defp write_json(filename, maps) do
    filename
    |> File.write!(
      maps
      |> Poison.encode!()
    )
  end
end
