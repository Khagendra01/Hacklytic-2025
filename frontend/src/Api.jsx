import { useEffect, useState } from "react";
import "./App.css";

function Api() {
  const [id, setId] = useState("5");
  const [query, setQuery] = useState("somequery");
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch(`http://127.0.0.1:8000/items/${id}?q=${query}`)
      .then((response) => response.json())
      .then((data) => setData(data))
      .catch((error) => console.error("Error fetching data:", error));
  }, [id, query]);

  return (
    <div className="App">
      <h1>FastAPI Response</h1>
      <label>
        ID:
        <input type="number" value={id} onChange={(e) => setId(e.target.value)} />
      </label>
      <label>
        Query:
        <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} />
      </label>
      {data ? <pre>{JSON.stringify(data, null, 2)}</pre> : <p>Loading...</p>}
    </div>
  );
}

export default Api;
