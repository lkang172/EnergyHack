import React from "react";
import { useEffect } from "react";
import hljs from "highlight.js";
import "highlight.js/styles/github-dark.css"; // Import a theme

import "../App.css";

const CodeInput = ({ onChange }) => {
  useEffect(() => {
    hljs.highlightAll();
  }, []);
  return (
    <>
      <textarea onChange={onChange} />
    </>
  );
};

export default CodeInput;
