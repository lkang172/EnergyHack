import 'bootstrap';

interface ParametersProps {

}

export default function ParameterInputContainer({} : ParametersProps) {
    return (
        <div>
            <input type='text' className='form-control'></input>
            <button type="button" className="btn btn-primary">Primary</button>
            <select name="gpus" className="btn btn-ptimary" >
                <option value="4090">4090</option>
            </select>
        </div>
    );
};