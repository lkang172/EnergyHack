import "../App.css";

const CodeInput = ({ onChange }: any) => {
  return (
    <>
      <textarea onChange={onChange} placeholder="//paste code here"></textarea>
    </>
  );
};

export default CodeInput;
